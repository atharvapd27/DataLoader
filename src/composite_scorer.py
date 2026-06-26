"""
Composite Scorer Module
Applies mathematical blending and JD penalty gates.

Fixes applied:
  - behavior_multiplier safety clamp (can't go negative)
  - location_match float-equality gate replaced with < 0.01
  - cv_without_nlp trap penalty wired in
  - open_to_work boost wired in
  - experience range soft penalty for < 5 years (JD sweet spot 5-9)
  - Pass 1 now returns base_score and behavior_multiplier separately
    so Pass 2 can blend with semantic score cleanly without double-applying multiplier
  - generate_reasoning overhauled: pulls specific facts, flags concerns,
    connects to JD requirements, varies by rank tier — passes Stage 4 checks
"""


class CompositeScorer:
    def __init__(self, semantic_ranker):
        self.semantic_ranker = semantic_ranker

    # ------------------------------------------------------------------
    # PRIMARY SCORING ENTRY POINT
    # ------------------------------------------------------------------

    def score_candidate(self, candidate_id, candidate, features, pass1_mode=False,
                        semantic_score=None):
        """
        Returns final float score in [0.0, 1.0].

        pass1_mode=True  → heuristics only, no semantic call (fast)
        pass1_mode=False → caller must supply semantic_score (pre-batched)
        """
        # 1. ABSOLUTE GATES
        if features['is_honeypot']:
            return 0.0
        if features['ghost_profile']:
            return 0.0
        if features['location_match'] < 0.01:
            return 0.0

        # 2. HEURISTIC BASE SCORE
        base_score = self._compute_base_score(features)

        # 3. BEHAVIORAL BONUS (additive, normalised to [0,1])
        behavior_bonus = self._compute_behavior_bonus(features)

        # 4. BLEND
        # Weights: sem 0.45 + heuristic 0.45 + behavior 0.10 = 1.0 max
        # Additive approach prevents score saturation that a multiplier causes
        if pass1_mode:
            # Pass 1: no semantic score yet — heuristics 0.90 + behavior 0.10
            final_score = (base_score * 0.90) + (behavior_bonus * 0.10)
        else:
            if semantic_score is None:
                raise ValueError("semantic_score must be provided in Pass 2 mode")
            final_score = (semantic_score * 0.45) + (base_score * 0.45) + (behavior_bonus * 0.10)

        return max(0.0, min(1.0, final_score))

    def get_base_and_bonus(self, features):
        """
        Used by rank.py Pass 2 to retrieve pre-computed components.
        Returns (base_score, behavior_bonus) for clean additive blend with semantic score.
        """
        if features['is_honeypot'] or features['ghost_profile'] or features['location_match'] < 0.01:
            return 0.0, 0.5  # 0.5 = neutral behavior bonus
        return self._compute_base_score(features), self._compute_behavior_bonus(features)

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _compute_base_score(self, features):
        """Heuristic base: tech signals minus JD trap penalties."""

        # Core technical foundation
        tech_score = (
            features['core_skills_depth']                    * 0.4 +
            min(features['eval_frameworks'], 1.0)            * 0.3 +
            min(features['production_builder'], 1.0)         * 0.3
        )

        # JD trap penalties
        penalty = (
            features['langchain_tourist']   * 0.40 +
            features['research_heavy']      * 0.40 +
            features['title_chaser']        * 0.30 +
            features['hands_off_architect'] * 0.50 +
            features['pure_consulting']     * 0.30 +
            features['keyword_stuffer']     * 0.80 +
            features['cv_without_nlp']      * 0.35   # new trap
        )

        # Soft penalty for outside JD experience sweet spot (5–9 yrs)
        # Honeypot gate already hard-kills < 3.5; this is 3.5–5 soft zone
        years = features.get('years_experience', 5)
        if years < 5:
            penalty += 0.15
        elif years > 12:
            penalty += 0.05   # slight over-qualification signal

        return max(0.01, tech_score - penalty)

    def _compute_behavior_bonus(self, features):
        """
        Behavioral signals normalised to [0.0, 1.0].
        Used as an additive third component (weight 0.10) rather than a
        multiplier — prevents score saturation at 1.0 for good candidates.

        Scoring:
          response_rate, notice_period, github, open_to_work, assessments
          each contribute independently; total is clipped to [0.0, 1.0].
        """
        bonus = 0.5  # neutral baseline

        # Response rate (0-1 signal, strong indicator of availability)
        rr = features['response_rate']
        if rr > 0.6:
            bonus += 0.20
        elif rr > 0.3:
            bonus += 0.05
        elif rr < 0.2:
            bonus -= 0.20

        # Notice period
        np_ = features['notice_period']
        if np_ <= 30:
            bonus += 0.15
        elif np_ <= 60:
            bonus += 0.05
        elif np_ > 90:
            bonus -= 0.15

        # GitHub activity (-1 = not provided → neutral)
        gh = features['github_score']
        if gh > 60:
            bonus += 0.10

        # Actively open to work
        if features['open_to_work']:
            bonus += 0.08

        # Skill assessment scores — one-time boost for proven platform competence
        assessments = features.get('assessments', {})
        relevant_kws = ['python', 'machine learning', 'sql', 'data', 'algorithm']
        for skill, ascore in assessments.items():
            if isinstance(ascore, (int, float)) and ascore > 80:
                if any(kw in skill.lower() for kw in relevant_kws):
                    bonus += 0.05
                    break  # one-time only

        # Interview completion rate — measures follow-through reliability
        # Low rate = scheduled interviews but didn't show = serious availability concern
        icr = features.get('interview_completion', 0.0)
        if icr > 0.8:
            bonus += 0.10
        elif icr > 0.5:
            bonus += 0.03
        elif 0 < icr < 0.4:
            bonus -= 0.15

        # Avg response time — fast responder = more reachable
        # 999 = not provided → neutral
        art = features.get('avg_response_hours', 999)
        if art < 24:
            bonus += 0.07
        elif art > 120:  # >5 days response time
            bonus -= 0.08

        # Profile completeness — incomplete profile = passive/not serious
        pc = features.get('profile_completeness', 0.0)
        if pc >= 80:
            bonus += 0.05
        elif pc < 40:
            bonus -= 0.05

        # Saved by recruiters — external market validation signal
        sbr = features.get('saved_by_recruiters', 0)
        if sbr >= 5:
            bonus += 0.06
        elif sbr >= 2:
            bonus += 0.03

        # Offer acceptance rate — -1 means no history (neutral)
        # Low rate with history = serial time waster
        oar = features.get('offer_acceptance')
        if oar is not None and oar != -1:
            if oar > 0.7:
                bonus += 0.04
            elif oar < 0.3:
                bonus -= 0.06

        return max(0.0, min(1.0, bonus))

    # ------------------------------------------------------------------
    # REASONING — Stage 4 compliant
    # ------------------------------------------------------------------

    def generate_reasoning(self, candidate, features, score, rank):
        """
        Stage 4 checks (from submission_spec.md):
          - Specific facts (years, company, named skills)
          - JD connection
          - Honest concerns
          - No hallucination (only uses what's in the profile)
          - Variation across candidates
          - Rank consistency (tone matches rank)

        This method pulls real facts from the candidate object.
        """
        profile  = candidate['profile']
        history  = candidate['career_history']
        skills   = candidate['skills']
        signals  = candidate['redrob_signals']

        title        = profile.get('current_title', 'Engineer')
        years        = profile.get('years_of_experience', 0)
        company      = profile.get('current_company', '')
        location     = profile.get('location', '')
        notice       = signals.get('notice_period_days', 90)
        resp_rate    = signals.get('recruiter_response_rate', 0.0)
        open_to_work = signals.get('open_to_work_flag', False)

        # Pull top 2 relevant skills by duration
        relevant_kws = ['python', 'embedding', 'vector', 'machine learning', 'pytorch',
                        'faiss', 'elasticsearch', 'retrieval', 'ranking', 'lora',
                        'sentence-transformer', 'opensearch', 'pinecone', 'qdrant']
        top_skills = [
            s for s in sorted(skills, key=lambda x: x.get('duration_months', 0), reverse=True)
            if any(kw in s['name'].lower() for kw in relevant_kws)
        ][:2]

        # Most recent employer context
        recent_company = history[0].get('company', '') if history else ''

        # ---- Assemble positives ----
        positives = []

        if features['eval_frameworks'] > 0:
            positives.append("demonstrates evaluation rigour (NDCG/MRR/A-B testing)")
        if features['production_builder'] > 0:
            positives.append("verifiable production shipping history")
        if features['core_skills_depth'] > 0.5:
            skill_str = ', '.join(
                f"{s['name']} ({s.get('duration_months', 0)}mo)" for s in top_skills
            ) if top_skills else 'core ML stack'
            positives.append(f"depth in {skill_str}")
        if resp_rate > 0.6:
            positives.append(f"high recruiter response rate ({int(resp_rate*100)}%)")
        if notice <= 30:
            positives.append(f"available quickly (notice: {notice}d)")
        if open_to_work:
            positives.append("actively open to work")

        # ---- Assemble concerns ----
        concerns = []

        if features['langchain_tourist']:
            concerns.append("LLM API usage without evident pre-LLM retrieval depth")
        if features['research_heavy']:
            concerns.append("research-heavy background without production deployment evidence")
        if features['title_chaser']:
            concerns.append("average tenure < 18 months across roles")
        if features['hands_off_architect']:
            concerns.append("senior title but no coding verbs in recent role description")
        if features['pure_consulting']:
            concerns.append("entire career in IT services firms")
        if features['cv_without_nlp']:
            concerns.append("CV/vision background without clear NLP or IR exposure")
        if notice > 90:
            concerns.append(f"long notice period ({notice}d)")
        if resp_rate < 0.2:
            concerns.append(f"low recruiter response rate ({int(resp_rate*100)}%)")
        if years < 5:
            concerns.append(f"below JD experience floor ({years} yrs vs 5-9 target)")
        if features['location_match'] < 0.9:
            concerns.append(f"requires relocation from {location}")

        # ---- Format by rank tier ----
        pos_str = '; '.join(positives[:2]) if positives else 'marginal technical signals'
        con_str = '; '.join(concerns[:2]) if concerns else 'no major red flags'

        if rank <= 10:
            tone = "Strong fit"
        elif rank <= 30:
            tone = "Good fit"
        elif rank <= 60:
            tone = "Moderate fit"
        else:
            tone = "Weak fit"

        company_str = recent_company or company
        reasoning = (
            f"{title} at {company_str} | {years} yrs | {tone} — "
            f"{pos_str}. "
            f"Concerns: {con_str}."
        )

        return reasoning[:300]