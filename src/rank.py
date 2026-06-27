#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranker
Two-pass pipeline: heuristic funnel → semantic re-ranker → top 100.

PRECOMPUTED MODE (recommended):
  Run precompute_embeddings.py once offline (~15-20 min).
  Then rank.py loads embeddings.npy at runtime — zero encoding overhead.
  This means ALL candidates surviving hard gates get semantic scoring,
  not just the top 3K from heuristics. Much better recall.

  python rank.py --candidates candidates.jsonl.gz --jd job_description.md \
                 --precomputed ./precomputed --out submission.csv

LIVE MODE (fallback, no precomputed files):
  python rank.py --candidates candidates.jsonl.gz --jd job_description.md \
                 --out submission.csv
"""

import argparse
import gzip
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from feature_extractor import FeatureExtractor
from semantic_ranker import SemanticRanker
from composite_scorer import CompositeScorer


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

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
            cid  = cand['candidate_id']
            candidates[cid] = cand
            ids_ordered.append(cid)
            if (i + 1) % 20_000 == 0:
                print(f"  Loaded {i+1:,} candidates...")
    print(f"  Total loaded: {len(candidates):,}")
    return candidates, ids_ordered


def load_jd(jd_path):
    if jd_path.endswith('.docx'):
        from docx import Document
        doc = Document(jd_path)
        return '\n'.join(p.text for p in doc.paragraphs)
    with open(jd_path, 'r') as f:
        return f.read()


def load_precomputed(precomputed_dir):
    """Load precomputed embeddings, candidate_ids, and JD embedding."""
    emb_path = os.path.join(precomputed_dir, 'embeddings.npy')
    ids_path = os.path.join(precomputed_dir, 'candidate_ids.json')
    jd_path  = os.path.join(precomputed_dir, 'jd_embedding.npy')

    for p in [emb_path, ids_path, jd_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing precomputed file: {p}")

    print(f"Loading precomputed embeddings from {precomputed_dir}...")
    embeddings   = np.load(emb_path)          # (N, D)
    jd_embedding = np.load(jd_path)           # (D,)
    with open(ids_path) as f:
        ids_ordered = json.load(f)

    print(f"  Embeddings: {embeddings.shape}, JD dim: {len(jd_embedding)}")
    return embeddings, ids_ordered, jd_embedding


# ---------------------------------------------------------------------------
# PRECOMPUTED MODE — uses full 100K semantic scores
# ---------------------------------------------------------------------------

def score_all_precomputed(candidates, ids_ordered, embeddings, jd_embedding, jd_text):
    print("\n" + "=" * 60)
    print("SCORING PIPELINE — PRECOMPUTED MODE")
    print("=" * 60)

    print("\n[1/4] Initialising scorers...")
    feature_extractor = FeatureExtractor(candidates, jd_text)
    composite_scorer  = CompositeScorer(semantic_ranker=None)  # no live encoding needed

    # Vectorised cosine similarity for ALL 100K in one shot
    print("\n[2/4] Computing semantic scores (vectorised, all 100K)...")
    t0 = datetime.now()
    jd_norm   = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-8)
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    unit_embs = embeddings / emb_norms
    sims      = unit_embs @ jd_norm                        # (N,)
    sem_scores = ((sims + 1.0) / 2.0).clip(0.0, 1.0)     # normalise to [0,1]
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  Done in {elapsed:.2f}s — semantic scores for {len(sem_scores):,} candidates")

    # Pass 1: heuristic gates + base scores for all 100K
    print(f"\n[3/4] Heuristic gates + scoring ({len(candidates):,} candidates)...")
    t1 = datetime.now()
    results = []
    total   = len(ids_ordered)

    for idx, cid in enumerate(ids_ordered):
        if (idx + 1) % 20_000 == 0:
            elapsed = (datetime.now() - t1).total_seconds()
            print(f"  [{idx+1:,}/{total:,}] {elapsed:.1f}s elapsed")

        try:
            cand     = candidates[cid]
            features = feature_extractor.extract_all_features(cid)

            # Hard gates — instant discard
            if features['is_honeypot']:       continue
            if features['ghost_profile']:     continue
            if features['location_match'] < 0.01: continue

            base_score, behavior_bonus = composite_scorer.get_base_and_bonus(features)
            sem_score  = float(sem_scores[idx])

            # Full additive blend with higher semantic weight (precomputed = higher quality)
            # sem=0.60, heuristic=0.30, behavior=0.10
            final_score = (sem_score * 0.60) + (base_score * 0.30) + (behavior_bonus * 0.10)
            final_score = round(max(0.0, min(1.0, final_score)), 6)

            results.append({
                'candidate_id':   cid,
                'score':          final_score,
                'base_score':     base_score,
                'behavior_bonus': behavior_bonus,
                'features':       features,
                'candidate':      cand,
            })
        except Exception:
            continue

    elapsed = (datetime.now() - t1).total_seconds()
    print(f"  Heuristic pass complete in {elapsed:.1f}s")
    print(f"  Candidates surviving gates: {len(results):,}")

    # Sort and take top 100
    print("\n[4/4] Finalising ranks...")
    results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    return results[:100], composite_scorer


# ---------------------------------------------------------------------------
# LIVE MODE — two-pass with on-the-fly encoding (fallback)
# ---------------------------------------------------------------------------

def score_all_live(candidates, ids_ordered, jd_text):
    print("\n" + "=" * 60)
    print("SCORING PIPELINE — LIVE MODE (no precomputed embeddings)")
    print("=" * 60)

    print("\n[1/4] Initialising scorers...")
    feature_extractor = FeatureExtractor(candidates, jd_text)
    semantic_ranker   = SemanticRanker(model_name='all-MiniLM-L6-v2')
    semantic_ranker.set_jd(jd_text)
    composite_scorer  = CompositeScorer(semantic_ranker)

    # Pass 1 — heuristic funnel over all 100K
    print(f"\n[2/4] Pass 1: Heuristic funnel ({len(candidates):,} candidates)...")
    t0 = datetime.now()
    pass1_results = []
    total = len(ids_ordered)

    for idx, cid in enumerate(ids_ordered):
        if (idx + 1) % 10_000 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  [{idx+1:,}/{total:,}] {100*(idx+1)/total:.1f}% — {elapsed:.1f}s elapsed")
        try:
            cand     = candidates[cid]
            features = feature_extractor.extract_all_features(cid)
            pass1_score = composite_scorer.score_candidate(cid, cand, features, pass1_mode=True)
            if pass1_score <= 0.0:
                continue
            base_score, behavior_bonus = composite_scorer.get_base_and_bonus(features)
            pass1_results.append({
                'candidate_id':   cid,
                'pass1_score':    pass1_score,
                'base_score':     base_score,
                'behavior_bonus': behavior_bonus,
                'features':       features,
                'candidate':      cand,
            })
        except Exception:
            continue

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  Pass 1 complete in {elapsed:.1f}s — {len(pass1_results):,} survivors")

    pass1_results.sort(key=lambda x: -x['pass1_score'])
    top_3000 = pass1_results[:3000]
    print(f"  Funnel → top {len(top_3000):,} → Pass 2")

    # Pass 2 — batched semantic re-ranking
    print(f"\n[3/4] Pass 2: Semantic re-ranking ({len(top_3000):,} candidates)...")
    t1 = datetime.now()
    top_ids      = [r['candidate_id'] for r in top_3000]
    sem_scores   = semantic_ranker.score_batch(candidates, top_ids)

    final_results = []
    for result, sem_score in zip(top_3000, sem_scores):
        base_score     = result['base_score']
        behavior_bonus = result['behavior_bonus']
        final_score    = (sem_score * 0.45) + (base_score * 0.45) + (behavior_bonus * 0.10)
        result['score'] = round(max(0.0, min(1.0, final_score)), 6)
        final_results.append(result)

    elapsed = (datetime.now() - t1).total_seconds()
    print(f"  Pass 2 complete in {elapsed:.1f}s")

    print("\n[4/4] Finalising ranks...")
    final_results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    return final_results[:100], composite_scorer


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def generate_output(top_100, composite_scorer, output_path):
    print(f"\nGenerating output CSV → {output_path}")
    rows = []
    for rank, result in enumerate(top_100, 1):
        reasoning = composite_scorer.generate_reasoning(
            result['candidate'], result['features'], result['score'], rank
        )
        rows.append({
            'candidate_id': result['candidate_id'],
            'rank':         rank,
            'score':        result['score'],
            'reasoning':    reasoning,
        })

    df = pd.DataFrame(rows, columns=['candidate_id', 'rank', 'score', 'reasoning'])
    df.to_csv(output_path, index=False)

    print(f"✓ Saved {output_path} ({len(df)} rows)")
    print("\nTop 10:")
    print(df[['candidate_id', 'rank', 'score']].head(10).to_string(index=False))
    return df


def validate_output(df):
    print("\nValidating submission format...")
    issues = []
    if len(df) != 100:
        issues.append(f"Row count: expected 100, got {len(df)}")
    for col in ['candidate_id', 'rank', 'score', 'reasoning']:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
    if set(df['rank'].tolist()) != set(range(1, 101)):
        issues.append("Ranks not exactly 1-100")
    scores = df['score'].tolist()
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            issues.append(f"Score not non-increasing: rank {i+1} ({scores[i]}) < rank {i+2} ({scores[i+1]})")
            break
    for i in range(len(scores) - 1):
        if scores[i] == scores[i + 1]:
            if df.iloc[i]['candidate_id'] > df.iloc[i+1]['candidate_id']:
                issues.append(f"Tie-break violation at ranks {i+1}/{i+2}")
    empty = df[df['reasoning'].isna() | (df['reasoning'] == '')].shape[0]
    if empty:
        issues.append(f"Empty reasoning in {empty} rows")
    if issues:
        print("✗ Validation FAILED:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    print("✓ Validation PASSED — ready for submission")
    return True


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Rank candidates for Redrob challenge')
    parser.add_argument('--candidates',   required=True, help='Path to candidates.jsonl or .jsonl.gz')
    parser.add_argument('--jd',           default='data/job_description.md', help='Path to job description')
    parser.add_argument('--out',          default='submission.csv', help='Output CSV path')
    parser.add_argument('--precomputed',  default=None,
                        help='Dir with precomputed embeddings.npy, candidate_ids.json, jd_embedding.npy. '
                             'If not provided, falls back to live encoding.')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("REDROB INTELLIGENT CANDIDATE RANKING")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    t_start = datetime.now()

    candidates, ids_ordered = load_candidates(args.candidates)
    jd_text = load_jd(args.jd)

    if args.precomputed:
        embeddings, emb_ids, jd_embedding = load_precomputed(args.precomputed)
        # Ensure ordering matches loaded candidates
        if emb_ids != ids_ordered:
            print("  WARNING: precomputed ID order differs from JSONL order — reindexing...")
            id_to_idx = {cid: i for i, cid in enumerate(emb_ids)}
            reordered = np.array([embeddings[id_to_idx[cid]] for cid in ids_ordered
                                  if cid in id_to_idx], dtype=np.float32)
            ids_ordered = [cid for cid in ids_ordered if cid in id_to_idx]
            embeddings  = reordered
        top_100, scorer = score_all_precomputed(
            candidates, ids_ordered, embeddings, jd_embedding, jd_text
        )
    else:
        top_100, scorer = score_all_live(candidates, ids_ordered, jd_text)

    df = generate_output(top_100, scorer, args.out)
    is_valid = validate_output(df)

    total_time = (datetime.now() - t_start).total_seconds()
    print(f"\nTotal pipeline time: {total_time:.1f}s")
    if total_time > 280:
        print("⚠  WARNING: approaching the 5-minute Stage 3 limit")

    if is_valid:
        print("\n" + "=" * 60)
        print("✓ READY FOR SUBMISSION")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n✗ Fix validation errors before submitting")
        sys.exit(1)


if __name__ == '__main__':
    main()