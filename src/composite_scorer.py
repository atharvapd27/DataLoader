"""
Composite Scorer Module
Combines all signals into final score with human-readable reasoning
"""

from datetime import datetime

class CompositeScorer:
    def __init__(self, semantic_ranker):
        self.semantic_ranker = semantic_ranker
        self.reference_date = datetime(2026, 6, 23)
    
    def score_candidate(self, candidate_id, candidate, features):
        """
        Final composite score combining all signals.
        
        Weights:
        - Semantic alignment: 25%
        - Systems depth: 25%
        - Career coherence: 20%
        - Behavioral availability: 20%
        - Anti-keyword-bias: 10%
        """
        
        # 1. SEMANTIC ALIGNMENT (25%)
        semantic_score = self.semantic_ranker.score_candidate(candidate)
        
        # 2. SYSTEMS DEPTH (25%) — CRITICAL
        systems_depth = (
            features['shipped_systems_signal'] * 0.5 +
            features['ml_depth_not_breadth'] * 0.3 +
            features['production_experience'] * 0.2
        )
        
        # 3. CAREER COHERENCE (20%)
        career_score = (
            features['career_coherence'] * 0.5 +
            features['product_company_exp'] * 0.4 +
            features['title_stability'] * 0.1
        )
        
        # 4. BEHAVIORAL AVAILABILITY (20%)
        behavioral = (
            features['recency_score'] * 0.35 +
            features['availability_score'] * 0.35 +
            features['engagement_score'] * 0.2 +
            features['github_signal'] * 0.1  # Bonus for GitHub activity
        )
        
        # 5. ANTI-KEYWORD-BIAS (10%)
        anti_bias_score = 1.0
        anti_bias_score += features['title_skill_mismatch']  # Negative penalty
        anti_bias_score += features['hidden_ai_skills'] * 0.05  # Positive: hidden depth
        anti_bias_score = max(0.0, min(1.0, anti_bias_score))
        
        # ============ HARD DISQUALIFIERS ============
        
        # Honeypot detection
        if features['is_honeypot_suspect']:
            return 0.0
        
        # Too junior for Senior role
        if candidate['profile']['years_of_experience'] < 3:
            return 0.1
        
        # Pure consulting background (soft penalize)
        if features['consulting_only']:
            # Not auto-disqualify but heavily penalize
            career_score *= 0.4
        
        # Very inactive (> 6 months)
        if features['inactive_flag']:
            behavioral *= 0.3
        
        # ============ BEHAVIORAL MULTIPLIER ============
        
        # Recent activity provides a multiplier boost
        recency = features['recency_score']
        
        if recency > 0.9:
            behavioral_multiplier = 1.12  # Recent = +12%
        elif recency > 0.7:
            behavioral_multiplier = 1.05
        elif recency > 0.3:
            behavioral_multiplier = 0.95
        else:
            behavioral_multiplier = 0.6  # Very inactive = -40%
        
        # Recruiter response rate multiplier
        response_rate = candidate['redrob_signals']['recruiter_response_rate']
        if response_rate > 0.7:
            response_multiplier = 1.08
        elif response_rate > 0.3:
            response_multiplier = 1.0
        else:
            response_multiplier = 0.75
        
        combined_multiplier = behavioral_multiplier * response_multiplier
        
        # ============ FINAL COMPOSITE ============
        
        score = (
            semantic_score * 0.25 +
            systems_depth * 0.25 +
            career_score * 0.20 +
            behavioral * 0.20 +
            anti_bias_score * 0.10
        )
        
        # Apply behavioral multiplier
        score = score * combined_multiplier
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def generate_reasoning(self, candidate, features, score, rank):
        """
        Generate human-readable reasoning for each ranked candidate.
        Stage 4 evaluation is harsh on generic/templated reasoning.
        Be specific and honest.
        """
        
        profile = candidate['profile']
        signals = candidate['redrob_signals']
        
        years = profile['years_of_experience']
        title = profile['current_title']
        location = profile['location']
        company = profile['current_company']
        
        # Collect specific reasons
        reasons = []
        
        # POSITIVE SIGNALS (specific)
        
        if features['shipped_systems_signal'] > 0.75:
            shipped_examples = self._extract_shipped_examples(candidate)
            if shipped_examples:
                reasons.append(f"shipped {shipped_examples}")
            else:
                reasons.append("clear track record of production systems")
        
        if features['core_skill_score'] > 0.7:
            core_skills = self._extract_core_skills(candidate)
            if core_skills:
                reasons.append(f"core skills: {core_skills}")
        
        if features['product_company_exp'] > 0.7:
            reasons.append("product company experience (not consulting)")
        
        if features['github_signal'] > 0.6:
            github_score = signals['github_activity_score']
            if github_score > 70:
                reasons.append(f"strong GitHub activity ({github_score})")
            else:
                reasons.append("active on GitHub")
        
        if features['skill_depth_score'] > 0.7:
            reasons.append("deep expertise in few areas (not breadth)")
        
        if features['recency_score'] > 0.85:
            days = self._days_since_active(signals['last_active_date'])
            reasons.append(f"recently active ({days} days ago)")
        
        if features['availability_score'] > 0.7:
            if signals['open_to_work_flag']:
                reasons.append("actively open to work")
            notice = signals['notice_period_days']
            if notice <= 30:
                reasons.append(f"short notice period ({notice} days)")
        
        if features['hidden_ai_skills'] > 0:
            reasons.append(f"additional ML depth not in skill list")
        
        if features['assessment_completion'] > 0.5:
            reasons.append("completed platform skill assessments")
        
        # CONCERNS (specific, honest)
        
        if features['recency_score'] < 0.3:
            days = self._days_since_active(signals['last_active_date'])
            reasons.append(f"⚠ inactive {days} days")
        
        if features['availability_score'] < 0.3:
            reasons.append("⚠ low availability signal")
        
        if features['consulting_only'] and features['shipped_systems_signal'] < 0.5:
            reasons.append("⚠ consulting background without clear shipped systems")
        
        if features['title_skill_mismatch'] < -0.2:
            reasons.append("⚠ title-skill misalignment")
        
        notice = signals['notice_period_days']
        if notice > 120:
            reasons.append(f"⚠ long notice period ({notice} days)")
        
        # BUILD REASONING STRING
        
        # Lead with position and experience
        reasoning = f"{title} ({years:.1f} yrs) @ {company}, {location}"
        
        # Add score tier
        if score > 0.85:
            reasoning += " | Strong fit"
        elif score > 0.70:
            reasoning += " | Good fit"
        elif score > 0.50:
            reasoning += " | Moderate fit"
        else:
            reasoning += " | Weak fit"
        
        # Add specific reasons (max 3 positive, 2 concerns)
        positive_reasons = [r for r in reasons if '⚠' not in r][:3]
        concern_reasons = [r for r in reasons if '⚠' in r][:2]
        
        if positive_reasons:
            reasoning += " | " + "; ".join(positive_reasons[:2])
        
        if concern_reasons:
            reasoning += " | " + "; ".join(concern_reasons[:1])
        
        # Truncate to reasonable length (avoid long strings)
        reasoning = reasoning[:250]
        
        return reasoning
    
    # ============ HELPER METHODS ============
    
    def _extract_shipped_examples(self, candidate):
        """Find specific examples of shipped systems from career history."""
        
        keywords = ['shipped', 'deployed', 'launched', 'production']
        
        for role in candidate['career_history']:
            desc = role['description'].lower()
            
            if any(kw in desc for kw in keywords):
                # Extract the system type if mentioned
                if 'ranking' in desc or 'retrieval' in desc or 'search' in desc:
                    return "ranking/retrieval systems"
                elif 'recommendation' in desc:
                    return "recommendation system"
                elif 'pipeline' in desc or 'ml' in desc:
                    return "ML/data systems"
                else:
                    return "production systems"
        
        return None
    
    def _extract_core_skills(self, candidate):
        """List core required skills found."""
        
        core_keywords = {
            'embeddings': ['embedding', 'sentence-transform'],
            'vector db': ['pinecone', 'weaviate', 'qdrant', 'milvus'],
            'python': ['python'],
            'evaluation': ['ndcg', 'mrr', 'evaluation']
        }
        
        found = []
        
        for skill in candidate['skills']:
            skill_lower = skill['name'].lower()
            
            for category, keywords in core_keywords.items():
                if any(kw in skill_lower for kw in keywords):
                    found.append(category)
                    break
        
        if found:
            return ', '.join(found[:3])
        
        return None
    
    def _days_since_active(self, last_active_date):
        """Calculate days since last active."""
        last_active = datetime.strptime(last_active_date, '%Y-%m-%d')
        days = (self.reference_date - last_active).days
        return days