"""
Semantic Ranker Module
Uses lightweight sentence-transformer for semantic matching.

Fixes applied:
  - score_candidate() kept for compatibility but Pass 2 should use score_batch()
  - score_batch() is now the primary Pass 2 entry point (called from main.py)
  - JD requirement phrases expanded and better aligned with actual JD language
  - jd_embedding stored as 1D vector; batch comparison uses vectorised matmul
  - Added explicit note: call set_jd() before any scoring
"""

import numpy as np


class SemanticRanker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"  Loading embedding model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("  ERROR: sentence-transformers not installed")
            print("  Install: pip install sentence-transformers")
            raise

        self.jd_embedding = None   # set by set_jd()
        self.model_name   = model_name

    # ------------------------------------------------------------------
    # JD SETUP (call once before any scoring)
    # ------------------------------------------------------------------

    def set_jd(self, jd_text):
        """Encode the job description into a single embedding vector."""
        print("  Embedding JD requirements...")
        jd_requirements = self._build_jd_text(jd_text)
        self.jd_embedding = self.model.encode(jd_requirements, convert_to_tensor=False)
        print(f"  JD embedding ready (dim={len(self.jd_embedding)})")

    def _build_jd_text(self, jd_text):
        """
        Build a focused semantic target from the JD.
        Combines curated requirement phrases with the first 1500 chars of
        the live JD so we catch any judge-side edits at Stage 3.
        """
        key_requirements = [
            # Hard requirements
            "Production experience embeddings-based retrieval systems real users",
            "Production vector databases hybrid search Pinecone Qdrant FAISS Elasticsearch",
            "Strong Python production code quality",
            "Evaluation frameworks ranking systems NDCG MRR MAP offline online A/B testing",

            # Systems-level signals
            "End-to-end ownership ranking search recommendation systems at scale",
            "Ship working ranker quickly demonstrably improves recruiter engagement metrics",
            "Deep technical depth embeddings retrieval ranking LLMs fine-tuning",
            "Scrappy product-engineering attitude willing to ship imperfect v1 and iterate",

            # Experience signals
            "Applied ML at product companies not pure research or consulting",
            "Built deployed production ML systems real users meaningful scale",
            "Set up evaluation infrastructure offline benchmarks online A/B testing recruiter feedback",

            # Soft positives
            "LLM fine-tuning LoRA QLoRA PEFT experience",
            "Learning to rank XGBoost neural LTR models",
            "Open source contributions AI ML space",
            "Active on platform in job market quick response rate",

            # Explicit disqualifiers (inverted phrasing helps push away bad candidates)
            "Not pure research academic lab without any production deployment",
            "Not recent LangChain tutorials OpenAI API calls without pre-LLM retrieval experience",
            "Not architecture director role without writing production code",
            "Not entire career TCS Infosys Wipro Accenture Cognizant consulting services",
            "Not computer vision speech robotics without NLP information retrieval experience",

            # Team / culture fit
            "Async-first written communication disagree openly decide quickly",
            "Plan to stay three plus years senior founding team engineer",
        ]

        core_text = ' '.join(key_requirements)
        # Append live JD text to catch any Stage 3 changes
        return core_text + "\n\n" + jd_text[:1500]

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------

    def score_batch(self, candidates, cand_ids):
        """
        PRIMARY PASS 2 METHOD.
        Encodes all candidates in a single vectorised batch operation.
        Returns list of float scores in [0.0, 1.0], same order as cand_ids.

        ~5-10x faster than calling score_candidate() in a loop.
        """
        if self.jd_embedding is None:
            raise RuntimeError("Call set_jd() before scoring.")

        texts      = [self._extract_candidate_text(candidates[cid]) for cid in cand_ids]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False
        )  # shape: (N, D)

        # Vectorised cosine similarity against single JD vector
        jd_norm  = self.jd_embedding / (np.linalg.norm(self.jd_embedding) + 1e-8)
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        unit_embs = embeddings / emb_norms
        similarities = unit_embs @ jd_norm  # shape: (N,)

        # Normalise from [-1, 1] to [0, 1]
        scores = (similarities + 1.0) / 2.0
        return [float(max(0.0, min(1.0, s))) for s in scores]

    def score_candidate(self, candidate):
        """
        Single-candidate scoring. Kept for compatibility / debugging.
        In production Pass 2, use score_batch() instead.
        """
        if self.jd_embedding is None:
            raise RuntimeError("Call set_jd() before scoring.")

        text       = self._extract_candidate_text(candidate)
        embedding  = self.model.encode(text, convert_to_tensor=False)
        similarity = np.dot(embedding, self.jd_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(self.jd_embedding) + 1e-8
        )
        return float(max(0.0, min(1.0, (similarity + 1.0) / 2.0)))

    # ------------------------------------------------------------------
    # CANDIDATE TEXT EXTRACTION
    # ------------------------------------------------------------------

    def _extract_candidate_text(self, candidate):
        """
        Convert a candidate record into a single searchable string.
        Weighted emphasis: skills with duration > summary > career desc > education.
        """
        parts = []

        profile = candidate.get('profile', {})

        # Headline and summary first — usually most signal-dense
        headline = profile.get('headline', '')
        summary  = profile.get('summary', '')
        if headline: parts.append(headline)
        if summary:  parts.append(summary)

        # Skills — emphasise longer-duration ones by repeating the name
        for s in candidate.get('skills', []):
            name     = s.get('name', '')
            prof     = s.get('proficiency', '')
            duration = s.get('duration_months', 0)
            if duration > 24:
                # Repeat to give semantic weight
                parts.append(f"{name} {name} ({prof}, {duration}mo)")
            else:
                parts.append(f"{name} ({prof})")

        # Career history — title, company, description
        for role in candidate.get('career_history', []):
            role_title = role.get('title', '')
            company    = role.get('company', '')
            desc       = role.get('description', '')
            if role_title or company:
                parts.append(f"{role_title} at {company}")
            if desc:
                parts.append(desc)

        # Education — field of study matters more than institution
        for edu in candidate.get('education', []):
            degree  = edu.get('degree', '')
            field   = edu.get('field_of_study', '')
            inst    = edu.get('institution', '')
            parts.append(f"{degree} {field} {inst}")

        return ' '.join(filter(None, parts))