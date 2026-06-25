"""
Semantic Ranker Module
Uses lightweight sentence-transformer for semantic matching
"""

import numpy as np

class SemanticRanker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize with lightweight model.
        all-MiniLM-L6-v2: 22M params, fast, reasonable quality
        """
        print(f"  Loading embedding model: {model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("  ERROR: sentence-transformers not installed")
            print("  Install with: pip install sentence-transformers")
            raise
        
        self.jd_embedding = None
        self.model_name = model_name
    
    def set_jd(self, jd_text):
        """Parse and embed the job description."""
        print("  Parsing JD...")
        
        # Extract key requirement sentences
        jd_requirements = self._extract_jd_requirements(jd_text)
        
        print(f"  Embedding JD ({len(jd_requirements)} tokens)...")
        self.jd_embedding = self.model.encode(jd_requirements, convert_to_tensor=False)
    
    def _extract_jd_requirements(self, jd_text):
        """
        Extract core requirements from JD.
        Focus on the TRUE intent, not keywords.
        """
        
        key_requirements = [
            # Core technical requirements
            "Production experience with embeddings-based retrieval systems deployed to real users",
            "Production experience with vector databases hybrid search infrastructure",
            "Strong Python code quality",
            "Hands-on experience designing evaluation frameworks for ranking systems NDCG MRR MAP",
            
            # Systems-level thinking
            "Own the intelligence layer ranking retrieval matching systems",
            "Ship working systems quickly demonstrably improves metrics",
            "Deep technical depth embeddings retrieval ranking LLMs fine-tuning",
            "Scrappy product-engineering attitude willing to ship week",
            
            # Experience signal
            "End-to-end ownership ranking search recommendation systems at scale",
            "Built production ML systems deployed to real users",
            "Evaluation infrastructure offline benchmarks online A/B testing",
            
            # Disqualifiers (inverted)
            "NOT pure research environments without production deployment",
            "NOT recent LangChain tutorials without substantial pre-LLM experience",
            "NOT architecture tech lead roles without writing production code",
            "NOT consulting services TCS Infosys Wipro entire career",
            
            # Team fit
            "Async-first writing oriented communication",
            "Disagree openly decide quickly",
            "Move fast break things learn from real users",
            "Plan to stay 3+ years growth opportunity"
        ]
        
        # Join all into one text
        jd_text_combined = ' '.join(key_requirements)
        
        return jd_text_combined
    
    def extract_candidate_text(self, candidate):
        """Convert candidate profile to searchable text."""
        
        parts = []
        
        # Profile headline + summary
        parts.append(candidate['profile']['headline'])
        parts.append(candidate['profile']['summary'])
        
        # Skills
        skills_text = ' '.join([
            f"{s['name']} ({s['proficiency']})"
            for s in candidate['skills']
        ])
        if skills_text:
            parts.append(skills_text)
        
        # Career history descriptions
        for role in candidate['career_history']:
            parts.append(f"{role['title']} at {role['company']}")
            parts.append(role['description'])
        
        # Education
        for edu in candidate.get('education', []):
            parts.append(f"{edu['degree']} in {edu['field_of_study']} from {edu['institution']}")
        
        return ' '.join(parts)
    
    def score_candidate(self, candidate):
        """
        Compute semantic similarity between candidate and JD.
        Returns 0-1 score.
        """
        
        # Extract candidate text
        cand_text = self.extract_candidate_text(candidate)
        
        # Embed candidate
        cand_embedding = self.model.encode(cand_text, convert_to_tensor=False)
        
        # Cosine similarity
        similarity = np.dot(cand_embedding, self.jd_embedding) / (
            np.linalg.norm(cand_embedding) * np.linalg.norm(self.jd_embedding) + 1e-8
        )
        
        # Normalize from [-1, 1] to [0, 1]
        normalized_score = (float(similarity) + 1.0) / 2.0
        
        return max(0.0, min(1.0, normalized_score))
    
    def score_batch(self, candidates, cand_ids):
        """Score multiple candidates efficiently."""
        
        texts = [self.extract_candidate_text(candidates[cid]) for cid in cand_ids]
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        
        scores = []
        for emb in embeddings:
            similarity = np.dot(emb, self.jd_embedding) / (
                np.linalg.norm(emb) * np.linalg.norm(self.jd_embedding) + 1e-8
            )
            normalized = (similarity + 1.0) / 2.0
            scores.append(max(0.0, min(1.0, normalized)))
        
        return scores