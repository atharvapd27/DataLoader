"""
Hyper-Optimized Feature Extraction Module
O(1) string blob processing with strict JD heuristic mapping.

Fixes applied:
  - Location gate expanded: Pune, Noida, Mumbai, Hyderabad, Delhi NCR / Gurgaon
  - Experience floor: < 3.5 years treated as disqualified
  - core_skills_depth keywords expanded to cover full JD "must have" list
  - eval_frameworks phrases loosened to catch natural language variants
  - CV/Speech/Robotics-without-NLP trap added
  - open_to_work_flag surfaced as a feature
  - github_score -1 (not provided) treated as neutral, not a penalty trigger
  - titles_blob dead code removed
"""

from datetime import datetime


class FeatureExtractor:
    def __init__(self, candidates, jd_text):
        self.candidates = candidates
        self.reference_date = datetime(2026, 6, 23)

        self.consulting_companies = {
            'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
            'deloitte', 'pwc', 'ey', 'kpmg'
        }

        # JD "things you absolutely need" — expanded to catch real profiles
        self.core_eval_frameworks = [
            'ndcg', 'mrr', 'map', 'a/b test', 'ab test',
            'offline eval', 'online eval', 'evaluation framework',
            'ranking eval', 'recall@', 'precision@', 'mean average precision'
        ]

        self.production_terms = [
            'shipped', 'deployed', 'production', 'launched',
            'scale', 'latency', 'real users', 'serving'
        ]

        self.research_terms = [
            'published', 'paper', 'academic', 'research paper', 'prototype'
        ]

        # JD "things you absolutely need" for retrieval/ranking systems
        self.core_skill_keywords = [
            'python', 'embedding', 'vector', 'machine learning', 'pytorch',
            'tensorflow', 'faiss', 'pinecone', 'qdrant', 'weaviate',
            'milvus', 'elasticsearch', 'opensearch', 'sentence-transformer',
            'retrieval', 'ranking', 'fine-tuning', 'lora', 'qlora',
            'bge', 'e5', 'reranking', 'hybrid search', 'bm25',
            'information retrieval', 'dense retrieval', 'sparse retrieval'
        ]

        # CV/Speech/Robotics — JD explicitly disqualifies these without NLP/IR
        self.cv_speech_terms = [
            'computer vision', 'object detection', 'image classification',
            'image segmentation', 'speech recognition', 'asr', 'tts',
            'robotics', 'autonomous driving', 'lidar', 'slam'
        ]

        self.nlp_ir_terms = [
            'nlp', 'retrieval', 'ranking', 'embeddings', 'search',
            'information retrieval', 'language model', 'text classification',
            'named entity', 'semantic search', 'question answering'
        ]

        # Delhi NCR covers Gurgaon, Noida, Faridabad, Greater Noida
        self.in_scope_cities = [
            'pune', 'noida', 'mumbai', 'hyderabad', 'delhi',
            'gurgaon', 'gurugram', 'faridabad', 'greater noida',
            'new delhi', 'ncr'
        ]

    # ------------------------------------------------------------------
    # PUBLIC: called per candidate in the Pass 1 loop
    # ------------------------------------------------------------------
    def extract_all_features(self, candidate_id):
        cand = self.candidates[candidate_id]
        profile  = cand['profile']
        history  = cand['career_history']
        skills   = cand['skills']
        signals  = cand['redrob_signals']

        # Build text blobs ONCE — O(1) per candidate from here on
        summary     = profile.get('summary', '') or ''
        career_desc = ' '.join(h.get('description', '') or '' for h in history)
        text_blob   = f"{summary} {career_desc}".lower()
        skills_blob = ' '.join(s['name'] for s in skills).lower()

        return {
            'candidate_id': candidate_id,

            # --- Hard gates ---
            'is_honeypot':     self._is_honeypot(profile, history, skills),
            'ghost_profile':   self._is_ghost_profile(signals),
            'location_match':  self._check_location(profile, signals),

            # --- JD trap penalties (each returns 0.0 or 1.0) ---
            'langchain_tourist':   self._langchain_tourist_trap(text_blob, skills_blob),
            'research_heavy':      self._research_vs_production(text_blob),
            'title_chaser':        self._title_chaser(history),
            'hands_off_architect': self._hands_off_architect(history),
            'pure_consulting':     self._pure_consulting(history),
            'keyword_stuffer':     self._keyword_stuffer_mismatch(profile, skills_blob),
            'cv_without_nlp':      self._cv_without_nlp(text_blob, skills_blob),

            # --- JD boosts (raw counts, capped in scorer) ---
            'eval_frameworks':   sum(0.25 for kw in self.core_eval_frameworks if kw in text_blob),
            'production_builder': sum(0.15 for kw in self.production_terms if kw in text_blob),
            'core_skills_depth': self._core_skill_depth(skills),

            # --- Behavioral signals ---
            # Use `or default` not `.get(key, default)` — guards against null JSON values
            # where the key exists but is set to None (e.g. "notice_period_days": null)
            'response_rate':          signals.get('recruiter_response_rate') or 0.0,
            'notice_period':          signals.get('notice_period_days') or 90,
            'github_score':           signals.get('github_activity_score') or -1,  # -1 = not provided
            'open_to_work':           1 if signals.get('open_to_work_flag', False) else 0,
            'assessments':            signals.get('skill_assessment_scores') or {},
            # Newly added signals from schema
            'interview_completion':   signals.get('interview_completion_rate') or 0.0,
            'avg_response_hours':     signals.get('avg_response_time_hours') or 999,
            'profile_completeness':   signals.get('profile_completeness_score') or 0.0,
            'saved_by_recruiters':    signals.get('saved_by_recruiters_30d') or 0,
            'offer_acceptance':       signals.get('offer_acceptance_rate'),  # None-safe: -1 is valid

            # --- Experience (used in scorer for soft range check) ---
            'years_experience': profile.get('years_of_experience', 0),
        }

    # ==================== HONEYPOTS & GHOSTS ====================

    def _is_honeypot(self, profile, history, skills):
        """Catches physically impossible profiles (Spec Section 7)."""
        years_exp = profile.get('years_of_experience', 0)

        # 1. Experience floor — JD says 5-9 years; < 3.5 is auto-disqualified
        if years_exp < 3.5:
            return True

        # 2. Expert skill with zero duration (classic honeypot signal)
        expert_no_duration = sum(
            1 for s in skills
            if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0
        )
        if expert_no_duration >= 5:
            return True

        # 3. Single role tenure longer than total claimed experience (+12mo buffer)
        claimed_exp_months = (years_exp * 12) + 12
        for role in history:
            if role.get('duration_months', 0) > claimed_exp_months:
                return True

        # 4. Principal title with fewer than 5 years
        title = profile.get('current_title', '').lower()
        if 'principal' in title and years_exp < 5:
            return True

        return False

    def _is_ghost_profile(self, signals):
        """Inactive > 6 months AND low response rate = effectively unavailable."""
        last_active_str = signals.get('last_active_date')
        if not last_active_str:
            return True

        try:
            last_active = datetime.strptime(last_active_str, '%Y-%m-%d')
        except ValueError:
            return True  # malformed date → treat as ghost
        days_inactive = (self.reference_date - last_active).days
        response_rate = signals.get('recruiter_response_rate', 0.0)

        return days_inactive > 180 and response_rate < 0.10

    # ==================== LOCATION ====================

    def _check_location(self, profile, signals):
        """
        JD: Pune / Noida preferred; Mumbai, Hyderabad, Delhi NCR explicitly welcome.
        Outside India: case-by-case; won't relocate + outside scope = 0.0.
        """
        loc     = profile.get('location', '').lower()
        country = profile.get('country', '').lower()

        if any(city in loc for city in self.in_scope_cities):
            return 1.0

        if signals.get('willing_to_relocate', False):
            # India candidates willing to move — strongly viable
            if country == 'india':
                return 0.8
            # Outside India + willing — JD says "case-by-case, no visa sponsor"
            return 0.4

        # Outside scope, not willing to relocate — hard zero
        return 0.0

    # ==================== JD TRAPS ====================

    def _langchain_tourist_trap(self, text_blob, skills_blob):
        """JD: Not people whose AI experience is just recent LangChain/OpenAI tutorials."""
        has_llm_api = 'langchain' in text_blob or 'openai' in text_blob
        has_deep_ml = (
            'pytorch' in text_blob or
            'tensorflow' in text_blob or
            'embeddings' in skills_blob or
            'sentence-transformer' in skills_blob or
            'fine-tuning' in text_blob
        )
        return 1.0 if (has_llm_api and not has_deep_ml) else 0.0

    def _research_vs_production(self, text_blob):
        """JD: Not pure research without production deployment."""
        research_count = sum(text_blob.count(kw) for kw in self.research_terms)
        prod_count     = sum(text_blob.count(kw) for kw in self.production_terms)
        return 1.0 if (research_count > 3 and prod_count == 0) else 0.0

    def _title_chaser(self, history):
        """JD: Switching companies every 1.5 years is a disqualifier."""
        if len(history) < 3:
            return 0.0
        avg_tenure = sum(h.get('duration_months', 0) for h in history) / len(history)
        return 1.0 if avg_tenure < 18 else 0.0

    def _hands_off_architect(self, history):
        """JD: Director/VP/Architect titles whose recent role has no hands-on coding."""
        if not history:
            return 0.0
        recent = history[0]
        title  = recent.get('title', '').lower()
        desc   = recent.get('description', '').lower()

        is_senior_title = any(kw in title for kw in ['architect', 'director', 'vp', 'head'])
        if is_senior_title:
            hands_on_verbs = ['coded', 'python', 'built', 'shipped', 'implemented',
                              'developed', 'wrote', 'engineered']
            if not any(kw in desc for kw in hands_on_verbs):
                return 1.0
        return 0.0

    def _pure_consulting(self, history):
        """JD: Entire career at TCS/Infosys/Wipro etc. is a disqualifier."""
        if not history:
            return 0.0
        consulting_count = sum(
            1 for role in history
            if any(kw in role.get('company', '').lower() for kw in self.consulting_companies)
        )
        return 1.0 if consulting_count == len(history) else 0.0

    def _keyword_stuffer_mismatch(self, profile, skills_blob):
        """Marketing/Sales/HR/Finance title + AI skills = keyword stuffer trap."""
        title       = profile.get('current_title', '').lower()
        is_non_tech = any(kw in title for kw in ['marketing', 'sales', 'hr', 'finance', 'recruiter'])
        has_ai      = 'machine learning' in skills_blob or 'llm' in skills_blob
        return 1.0 if (is_non_tech and has_ai) else 0.0

    def _cv_without_nlp(self, text_blob, skills_blob):
        """
        JD: CV/Speech/Robotics specialists without NLP/IR exposure.
        Combined blob check — catches profile + skill mentions.
        """
        combined    = text_blob + ' ' + skills_blob
        cv_count    = sum(combined.count(t) for t in self.cv_speech_terms)
        has_nlp_ir  = any(t in combined for t in self.nlp_ir_terms)
        return 1.0 if (cv_count >= 2 and not has_nlp_ir) else 0.0

    # ==================== CORE DEPTH ====================

    def _core_skill_depth(self, skills):
        """
        Validates JD must-have skills by proficiency, duration, and endorsements.
        Expanded keyword list covers full JD retrieval/ranking/vector DB stack.
        """
        score = 0.0
        for s in skills:
            name = s['name'].lower()
            if any(kw in name for kw in self.core_skill_keywords):
                if s.get('proficiency') in ['advanced', 'expert']:
                    score += 0.2
                if s.get('duration_months', 0) > 24:
                    score += 0.2
                if s.get('endorsements', 0) > 10:
                    score += 0.1
        return min(score, 1.0)