"""
Feature Extraction Module
Extracts comprehensive features from candidate profiles for scoring
"""

from datetime import datetime
from collections import Counter

class FeatureExtractor:
    def __init__(self, candidates, jd_text):
        self.candidates = candidates
        self.jd_text = jd_text
        self.reference_date = datetime(2026, 6, 23)  # Challenge reference date
        
        # Skill categories
        self.ai_keywords = {
            'machine learning', 'deep learning', 'nlp', 'llm', 'embeddings',
            'vector db', 'retrieval', 'ranking', 'transformers', 'pytorch',
            'tensorflow', 'keras', 'rag', 'fine-tun', 'lora', 'peft',
            'semantic search', 'vector search', 'pinecone', 'weaviate',
            'milvus', 'qdrant', 'elasticsearch', 'opensearch', 'langchain'
        }
        
        self.product_companies = {
            'google', 'facebook', 'meta', 'amazon', 'microsoft', 'apple',
            'netflix', 'uber', 'airbnb', 'spotify', 'linkedin', 'twitter',
            'stripe', 'shopify', 'databricks', 'anthropic', 'openai'
        }
        
        self.consulting_companies = {
            'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
            'deloitte', 'pwc', 'ey', 'kpmg'
        }
    
    def extract_all_features(self, candidate_id):
        """Extract comprehensive feature set."""
        cand = self.candidates[candidate_id]
        
        features = {
            'candidate_id': candidate_id,
            
            # Basic profile
            'years_exp': cand['profile']['years_of_experience'],
            'current_title': cand['profile']['current_title'],
            'current_company': cand['profile']['current_company'],
            'location': cand['profile']['location'],
            
            # Skill features
            'skill_count': len(cand['skills']),
            'ai_skill_count': self._count_ai_skills(cand),
            'core_skill_score': self._score_core_skills(cand),
            'skill_depth_score': self._score_skill_depth(cand),
            'hidden_ai_skills': self._detect_hidden_skills(cand),
            
            # Career features
            'avg_tenure_months': self._avg_tenure(cand),
            'career_coherence': self._career_coherence(cand),
            'product_company_exp': self._product_company_exp(cand),
            'consulting_only': self._consulting_company_exp(cand),
            'title_stability': self._title_stability(cand),
            
            # Systems depth (CRITICAL)
            'shipped_systems_signal': self._shipped_systems_signal(cand),
            'ml_depth_not_breadth': self._ml_depth_not_breadth(cand),
            'production_experience': self._production_experience(cand),
            
            # Behavioral signals
            'recency_score': self._recency_score(cand),
            'availability_score': self._availability_score(cand),
            'engagement_score': self._engagement_score(cand),
            'github_signal': self._github_signal(cand),
            'assessment_completion': self._assessment_completion(cand),
            
            # Red flags
            'is_honeypot_suspect': self._is_honeypot_suspect(cand),
            'title_skill_mismatch': self._title_skill_mismatch(cand),
            'inactive_flag': self._inactive_flag(cand),
        }
        
        return features
    
    # ============ SKILL FEATURES ============
    
    def _count_ai_skills(self, cand):
        """Count AI/ML related skills."""
        count = 0
        for skill in cand['skills']:
            if any(kw in skill['name'].lower() for kw in self.ai_keywords):
                count += 1
        return count
    
    def _score_core_skills(self, cand):
        """Score presence of CORE required skills (not just any AI skill)."""
        core_requirements = {
            'embeddings': ['embeddings', 'sentence-transformer', 'bge', 'e5', 'vector embedding'],
            'vector_db': ['pinecone', 'weaviate', 'qdrant', 'milvus', 'vector', 'faiss', 'opensearch'],
            'python': ['python'],
            'evaluation': ['ndcg', 'mrr', 'map', 'evaluation', 'ranking metrics']
        }
        
        skills_lower = [s['name'].lower() for s in cand['skills']]
        score = 0.0
        weight = {'embeddings': 0.3, 'vector_db': 0.3, 'python': 0.2, 'evaluation': 0.2}
        
        for req, keywords in core_requirements.items():
            found = any(any(kw in skill for kw in keywords) for skill in skills_lower)
            if found:
                score += weight[req]
        
        return min(score, 1.0)
    
    def _score_skill_depth(self, cand):
        """Score: deep expertise > breadth."""
        skills = cand['skills']
        
        if not skills:
            return 0.0
        
        # Score 1: Ratio of advanced skills
        advanced_count = sum(1 for s in skills if s['proficiency'] in ['advanced', 'expert'])
        advanced_ratio = advanced_count / len(skills)
        
        # Score 2: Average endorsements (signals real expertise)
        endorsements = [s['endorsements'] for s in skills]
        avg_endorsements = sum(endorsements) / len(endorsements)
        endorsement_score = min(avg_endorsements / 30, 1.0)
        
        # Score 3: Deep duration in key skills
        ai_skills = [s for s in skills if any(kw in s['name'].lower() for kw in self.ai_keywords)]
        if ai_skills:
            deep_ai = [s for s in ai_skills if s['duration_months'] > 24]
            deep_ratio = len(deep_ai) / len(ai_skills)
        else:
            deep_ratio = 0.0
        
        return (advanced_ratio * 0.3 + endorsement_score * 0.3 + deep_ratio * 0.4)
    
    def _detect_hidden_skills(self, cand):
        """Detect skills not explicitly listed but present in work history."""
        descriptions = '\n'.join([h['description'].lower() for h in cand['career_history']])
        
        hidden_skills = 0
        
        skill_indicators = {
            'embeddings': ['embedding', 'semantic search', 'vector space'],
            'retrieval': ['retrieval', 'search', 'ranking', 'recommendation system'],
            'evaluation': ['metric', 'ndcg', 'accuracy evaluation', 'benchmark'],
            'llms': ['large language model', 'llm', 'gpt', 'bert', 'transformer'],
            'production_ml': ['production', 'deployed', 'shipped', 'live model'],
        }
        
        for skill, indicators in skill_indicators.items():
            if any(ind in descriptions for ind in indicators):
                if not any(skill in s['name'].lower() for s in cand['skills']):
                    hidden_skills += 1
        
        return hidden_skills
    
    # ============ CAREER FEATURES ============
    
    def _avg_tenure(self, cand):
        """Average tenure in months."""
        history = cand['career_history']
        if not history:
            return 0
        return sum(h['duration_months'] for h in history) / len(history)
    
    def _career_coherence(self, cand):
        """Score: consistent growth > job hopping."""
        history = cand['career_history']
        if len(history) < 2:
            return 1.0
        
        # Penalize short roles (< 12 months)
        short_roles = sum(1 for h in history if h['duration_months'] < 12)
        hopping_penalty = (short_roles / len(history)) * 0.4
        
        # Reward progression
        titles = [h['title'].lower() for h in history]
        progression_keywords = ['senior', 'lead', 'principal', 'staff', 'manager', 'architect']
        has_progression = any(any(kw in t for kw in progression_keywords) for t in titles)
        progression_bonus = 0.2 if has_progression else 0
        
        return max(0.3, 1.0 - hopping_penalty + progression_bonus)
    
    def _product_company_exp(self, cand):
        """Fraction of career at product/tech companies."""
        history = cand['career_history']
        product_months = 0
        total_months = 0
        
        for role in history:
            company = role['company'].lower()
            total_months += role['duration_months']
            
            is_consulting = any(kw in company for kw in self.consulting_companies)
            is_product = (
                any(kw in company for kw in self.product_companies) or
                role['industry'] in ['Technology', 'Software', 'SaaS', 'FinTech']
            )
            
            if is_product and not is_consulting:
                product_months += role['duration_months']
        
        return product_months / max(total_months, 1)
    
    def _consulting_company_exp(self, cand):
        """Check if ENTIRE career is at consulting companies."""
        history = cand['career_history']
        if not history:
            return False
        
        consulting_count = sum(
            1 for role in history
            if any(kw in role['company'].lower() for kw in self.consulting_companies)
        )
        
        return consulting_count == len(history)
    
    def _title_stability(self, cand):
        """Score title progression (penalize erratic changes)."""
        history = cand['career_history']
        if len(history) < 2:
            return 1.0
        
        title_changes = 0
        for i in range(len(history) - 1):
            prev = history[i]['title'].lower()
            curr = history[i+1]['title'].lower()
            
            if not self._titles_in_same_domain(prev, curr):
                title_changes += 1
        
        return 1.0 - (title_changes / max(len(history) - 1, 1)) * 0.3
    
    def _titles_in_same_domain(self, t1, t2):
        """Check if two titles are in the same career domain."""
        domains = {
            'engineer': ['engineer', 'developer', 'architect', 'cto'],
            'ml': ['ml', 'ai', 'data scientist', 'machine learning', 'research scientist'],
            'analyst': ['analyst', 'analytics'],
            'manager': ['manager', 'lead', 'director', 'vp', 'head']
        }
        
        for domain, keywords in domains.items():
            if all(any(kw in t for kw in keywords) for t in [t1, t2]):
                return True
        
        return False
    
    # ============ SYSTEMS DEPTH (CRITICAL) ============
    
    def _shipped_systems_signal(self, cand):
        """CRITICAL: Does their work history show SHIPPED systems?"""
        descriptions = '\n'.join([h['description'] for h in cand['career_history']])
        descriptions_lower = descriptions.lower()
        
        # Positive signals (higher weight)
        shipped_signals = {
            'shipped': 2.0, 'deployed': 2.0, 'production': 2.0, 'launched': 2.0,
            'built end-to-end': 2.5, 'owned': 1.5, 'architected': 1.5,
            'built': 1.2, 'designed': 1.0, 'implemented': 1.0,
        }
        
        # Negative signals (penalize non-production)
        research_signals = {
            'published paper': -0.8, 'research paper': -0.8, 'academic': -0.5,
            'experimented': -0.3, 'prototype': -0.3, 'poc': -0.3
        }
        
        score = 0.5  # Baseline
        
        for signal, weight in shipped_signals.items():
            count = descriptions_lower.count(signal)
            score += count * weight * 0.08
        
        for signal, weight in research_signals.items():
            count = descriptions_lower.count(signal)
            score += count * weight * 0.08
        
        return min(max(score, 0.1), 1.0)
    
    def _ml_depth_not_breadth(self, cand):
        """ML depth in specific areas > LangChain tutorial breadth."""
        summary = cand['profile']['summary'].lower()
        descriptions = '\n'.join([h['description'].lower() for h in cand['career_history']])
        text = summary + ' ' + descriptions
        
        depth_signals = {
            'embeddings': 1.5, 'vector': 1.5, 'retrieval': 1.5,
            'ranking': 1.5, 'fine-tuning': 1.5, 'fine-tun': 1.5,
            'evaluation': 1.0, 'ndcg': 1.0, 'mrr': 1.0,
        }
        
        breadth_penalties = {
            'langchain': -0.5, 'tutorial': -0.3, 'demo': -0.2,
            'side project': -0.2, 'pet project': -0.2
        }
        
        score = 0.5
        
        for signal, weight in depth_signals.items():
            if signal in text:
                score += weight * 0.1
        
        for penalty, weight in breadth_penalties.items():
            if penalty in text:
                score += weight * 0.1
        
        return min(max(score, 0.1), 1.0)
    
    def _production_experience(self, cand):
        """Score production system experience."""
        descriptions = '\n'.join([h['description'].lower() for h in cand['career_history']])
        
        prod_keywords = [
            'production', 'deployed', 'shipped', 'live', 'user-facing',
            'scaled', 'millions of users', 'latency', 'throughput'
        ]
        
        found = sum(1 for kw in prod_keywords if kw in descriptions)
        
        return min(found / 3, 1.0)
    
    # ============ BEHAVIORAL SIGNALS ============
    
    def _recency_score(self, cand):
        """How recently has candidate been active?"""
        last_active = datetime.strptime(cand['redrob_signals']['last_active_date'], '%Y-%m-%d')
        days_inactive = (self.reference_date - last_active).days
        
        if days_inactive < 7:
            return 1.0
        elif days_inactive < 30:
            return 0.9 - (days_inactive - 7) / 23 * 0.1
        elif days_inactive < 90:
            return 0.8 - (days_inactive - 30) / 60 * 0.3
        else:
            return max(0.3, 0.5 - (days_inactive - 90) / 365 * 0.2)
    
    def _availability_score(self, cand):
        """Composite availability: open to work + response rate + notice."""
        signals = cand['redrob_signals']
        
        score = 0.0
        
        # Open to work flag
        if signals['open_to_work_flag']:
            score += 0.4
        
        # Recruiter response rate
        score += signals['recruiter_response_rate'] * 0.3
        
        # Notice period (< 30 days is best)
        notice = signals['notice_period_days']
        if notice <= 30:
            score += 0.3
        elif notice <= 60:
            score += 0.15
        # else: 0
        
        return min(score, 1.0)
    
    def _engagement_score(self, cand):
        """Recruiter search views + interview engagement."""
        signals = cand['redrob_signals']
        
        score = 0.0
        
        # Search appearances
        views = min(signals['search_appearance_30d'], 20)
        score += (views / 20) * 0.3
        
        # Saved by recruiters
        saved = min(signals['saved_by_recruiters_30d'], 10)
        score += (saved / 10) * 0.3
        
        # Interview completion rate
        score += signals['interview_completion_rate'] * 0.4
        
        return min(score, 1.0)
    
    def _github_signal(self, cand):
        """GitHub activity score (0-100)."""
        score = cand['redrob_signals']['github_activity_score']
        
        if score < 0:  # No GitHub
            return 0.2
        elif score < 30:
            return 0.4
        elif score < 60:
            return 0.7
        else:
            return 0.95
    
    def _assessment_completion(self, cand):
        """Skill assessments completed on platform."""
        signals = cand['redrob_signals']
        assessments = signals['skill_assessment_scores']
        
        if not assessments:
            return 0.0
        
        scores = list(assessments.values())
        avg_score = sum(scores) / len(scores)
        
        # High assessment count + high average
        return min(len(assessments) / 5 * 0.5 + (avg_score / 100) * 0.5, 1.0)
    
    # ============ RED FLAGS ============
    
    def _is_honeypot_suspect(self, cand):
        """Detect impossible profiles."""
        profile = cand['profile']
        
        # Check 1: Claims long experience at young company
        history = cand['career_history']
        for role in history:
            if role['is_current']:
                company_name = role['company'].lower()
                
                # Startup companies
                if 'startup' in company_name or 'series' in company_name:
                    if profile['years_of_experience'] > 8 and role['duration_months'] > 36:
                        return True
        
        # Check 2: Many expert skills with no duration
        skills = cand['skills']
        expert_no_duration = sum(
            1 for s in skills
            if s['proficiency'] == 'expert' and s['duration_months'] < 1
        )
        if expert_no_duration > 5:
            return True
        
        # Check 3: Title-experience mismatch
        title = profile['current_title'].lower()
        years = profile['years_of_experience']
        
        if 'principal' in title and years < 7:
            return True
        if 'entry' in title and years > 5:
            return True
        
        return False
    
    def _title_skill_mismatch(self, cand):
        """Penalize: many AI skills but non-technical title."""
        title = cand['profile']['current_title'].lower()
        ai_count = self._count_ai_skills(cand)
        
        bad_titles = {
            'manager': -0.4, 'director': -0.3, 'vp': -0.4,
            'marketing': -0.6, 'sales': -0.6, 'hr': -0.6,
            'finance': -0.5, 'legal': -0.5, 'recruiter': -0.5
        }
        
        for bad_title, penalty in bad_titles.items():
            if bad_title in title and ai_count > 5:
                return penalty
        
        return 0.0
    
    def _inactive_flag(self, cand):
        """Red flag: very inactive."""
        signals = cand['redrob_signals']
        
        last_active = datetime.strptime(signals['last_active_date'], '%Y-%m-%d')
        days_inactive = (self.reference_date - last_active).days
        
        return days_inactive > 180  # More than 6 months