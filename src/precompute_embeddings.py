#!/usr/bin/env python3
"""
Precompute Embeddings Script — Run ONCE offline before submission.

Encodes all 100K candidate profiles + the JD using all-mpnet-base-v2
(higher quality than MiniLM, worth the offline time).

Produces:
  embeddings.npy        — shape (N, 768), float32, candidate embeddings in order
  candidate_ids.json    — ordered list of candidate_ids matching rows in embeddings.npy
  jd_embedding.npy      — shape (768,), the JD embedding

Runtime: ~15-20 minutes on CPU for 100K candidates.
These files are loaded by rank.py at scoring time — zero encoding overhead during ranking.

Usage:
  python precompute_embeddings.py \
    --candidates ./candidates.jsonl.gz \
    --jd ./data/job_description.md \
    --out_dir ./precomputed
"""

import argparse
import gzip
import json
import os
from datetime import datetime

import numpy as np


def load_candidates(jsonl_path):
    opener = gzip.open if jsonl_path.endswith('.gz') else open
    candidates = {}
    ids_ordered = []
    print(f"Loading candidates from {jsonl_path}...")
    with opener(jsonl_path, 'rt') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand['candidate_id']
            candidates[cid] = cand
            ids_ordered.append(cid)
            if (i + 1) % 20_000 == 0:
                print(f"  Loaded {i+1:,}...")
    print(f"  Total: {len(candidates):,}")
    return candidates, ids_ordered


def load_jd(jd_path):
    if jd_path.endswith('.docx'):
        from docx import Document
        doc = Document(jd_path)
        return '\n'.join(p.text for p in doc.paragraphs)
    with open(jd_path, 'r') as f:
        return f.read()


def extract_candidate_text(candidate):
    """Same extraction logic as semantic_ranker.py — must stay in sync."""
    parts = []
    profile = candidate.get('profile', {})

    headline = profile.get('headline', '')
    summary  = profile.get('summary', '')
    if headline: parts.append(headline)
    if summary:  parts.append(summary)

    for s in candidate.get('skills', []):
        name     = s.get('name', '')
        prof     = s.get('proficiency', '')
        duration = s.get('duration_months', 0)
        if duration > 24:
            parts.append(f"{name} {name} ({prof}, {duration}mo)")
        else:
            parts.append(f"{name} ({prof})")

    for role in candidate.get('career_history', []):
        role_title = role.get('title', '')
        company    = role.get('company', '')
        desc       = role.get('description', '')
        if role_title or company:
            parts.append(f"{role_title} at {company}")
        if desc:
            parts.append(desc)

    for edu in candidate.get('education', []):
        parts.append(f"{edu.get('degree','')} {edu.get('field_of_study','')} {edu.get('institution','')}")

    return ' '.join(filter(None, parts))


def build_jd_text(jd_text):
    """Same JD text as semantic_ranker.py — must stay in sync."""
    key_requirements = [
        "Production experience embeddings-based retrieval systems real users",
        "Production vector databases hybrid search Pinecone Qdrant FAISS Elasticsearch",
        "Strong Python production code quality",
        "Evaluation frameworks ranking systems NDCG MRR MAP offline online A/B testing",
        "End-to-end ownership ranking search recommendation systems at scale",
        "Ship working ranker quickly demonstrably improves recruiter engagement metrics",
        "Deep technical depth embeddings retrieval ranking LLMs fine-tuning",
        "Scrappy product-engineering attitude willing to ship imperfect v1 and iterate",
        "Applied ML at product companies not pure research or consulting",
        "Built deployed production ML systems real users meaningful scale",
        "Set up evaluation infrastructure offline benchmarks online A/B testing recruiter feedback",
        "LLM fine-tuning LoRA QLoRA PEFT experience",
        "Learning to rank XGBoost neural LTR models",
        "Open source contributions AI ML space",
        "Active on platform in job market quick response rate",
        "Not pure research academic lab without any production deployment",
        "Not recent LangChain tutorials OpenAI API calls without pre-LLM retrieval experience",
        "Not architecture director role without writing production code",
        "Not entire career TCS Infosys Wipro Accenture Cognizant consulting services",
        "Not computer vision speech robotics without NLP information retrieval experience",
        "Async-first written communication disagree openly decide quickly",
        "Plan to stay three plus years senior founding team engineer",
    ]
    return ' '.join(key_requirements) + "\n\n" + jd_text[:1500]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--jd',         required=True)
    parser.add_argument('--out_dir',    default='./precomputed')
    parser.add_argument('--model',      default='all-MiniLM-L6-v2',
                        help='Use all-mpnet-base-v2 for best quality (dim=768) '
                             'or all-MiniLM-L6-v2 for speed (dim=384)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    candidates, ids_ordered = load_candidates(args.candidates)
    jd_text = load_jd(args.jd)

    # Init model
    print(f"\nLoading model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    # Embed JD
    print("Embedding JD...")
    jd_combined   = build_jd_text(jd_text)
    jd_embedding  = model.encode(jd_combined, convert_to_tensor=False, show_progress_bar=False)
    jd_path       = os.path.join(args.out_dir, 'jd_embedding.npy')
    np.save(jd_path, jd_embedding.astype(np.float32))
    print(f"  Saved JD embedding → {jd_path} (dim={len(jd_embedding)})")

    # Embed all candidates in batches
    print(f"\nEncoding {len(ids_ordered):,} candidates (this takes ~15-20 min on CPU)...")
    t0 = datetime.now()

    texts = [extract_candidate_text(candidates[cid]) for cid in ids_ordered]

    BATCH = 32
    all_embeddings = []
    total = len(texts)

    for start in range(0, total, BATCH):
        batch = texts[start:start + BATCH]
        embs  = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        all_embeddings.append(embs.astype(np.float32))

        done = min(start + BATCH, total)
        if done % 2_000 == 0 or done == total:
            elapsed = (datetime.now() - t0).total_seconds()
            eta     = (elapsed / done) * (total - done)
            print(f"  [{done:,}/{total:,}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    embeddings_matrix = np.vstack(all_embeddings)  # shape: (N, D)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nEncoding complete in {elapsed:.0f}s")
    print(f"Embeddings shape: {embeddings_matrix.shape}")

    # Save
    emb_path = os.path.join(args.out_dir, 'embeddings.npy')
    ids_path = os.path.join(args.out_dir, 'candidate_ids.json')

    np.save(emb_path, embeddings_matrix)
    with open(ids_path, 'w') as f:
        json.dump(ids_ordered, f)

    print(f"\nSaved:")
    print(f"  {emb_path}  ({embeddings_matrix.nbytes / 1e6:.1f} MB)")
    print(f"  {ids_path}")
    print(f"  {jd_path}")
    print(f"\nRun rank.py with --precomputed {args.out_dir} to use these.")


if __name__ == '__main__':
    main()