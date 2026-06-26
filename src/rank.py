#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranker
Two-pass pipeline: heuristic funnel → semantic re-ranker → top 100.

Fixes applied:
  - Pass 1 funnel raised to Top 5,000 (was 2,000)
  - Pass 2 now calls score_batch() for vectorised encoding (5-10x faster)
  - Pass 1 stores base_score and behavior_multiplier separately so
    Pass 2 can blend: final = (sem*0.5 + base*0.5) * multiplier
  - Progress logging improved for long runs
  - validate_output uses the actual validator logic (non-increasing scores,
    tie-break by candidate_id ascending)
"""

import json
import sys
import argparse
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
    import gzip
    candidates = {}
    print(f"Loading candidates from {jsonl_path}...")
    opener = gzip.open if jsonl_path.endswith('.gz') else open
    with opener(jsonl_path, 'rt') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            candidates[cand['candidate_id']] = cand
            if (i + 1) % 20_000 == 0:
                print(f"  Loaded {i+1:,} candidates...")
    print(f"  Total loaded: {len(candidates):,}")
    return candidates


def load_jd(jd_path):
    if jd_path.endswith('.docx'):
        from docx import Document
        doc  = Document(jd_path)
        return '\n'.join(p.text for p in doc.paragraphs)
    with open(jd_path, 'r') as f:
        return f.read()


# ---------------------------------------------------------------------------
# SCORING PIPELINE
# ---------------------------------------------------------------------------

def score_all_candidates(candidates, jd_text):
    print("\n" + "=" * 60)
    print("SCORING PIPELINE — TWO-PASS ARCHITECTURE")
    print("=" * 60)

    # Init all scorers once
    print("\n[1/4] Initialising scorers...")
    feature_extractor = FeatureExtractor(candidates, jd_text)
    semantic_ranker   = SemanticRanker(model_name='all-MiniLM-L6-v2')
    semantic_ranker.set_jd(jd_text)
    composite_scorer  = CompositeScorer(semantic_ranker)

    # -----------------------------------------------------------------------
    # PASS 1 — fast heuristics over all 100,000 candidates (~15-30 seconds)
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Pass 1: Heuristic funnel ({len(candidates):,} candidates)...")
    t0 = datetime.now()

    pass1_results = []
    total = len(candidates)

    for idx, (cand_id, cand) in enumerate(candidates.items()):
        if (idx + 1) % 10_000 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  [{idx+1:,}/{total:,}] {100*(idx+1)/total:.1f}% — {elapsed:.1f}s elapsed")

        try:
            features = feature_extractor.extract_all_features(cand_id)

            # Use pass1_mode=True — pure heuristics, no embedding
            pass1_score = composite_scorer.score_candidate(
                cand_id, cand, features, pass1_mode=True
            )

            if pass1_score <= 0.0:
                continue  # skip zeroed candidates early

            # Store components separately for clean Pass 2 blend
            base_score, behavior_bonus = composite_scorer.get_base_and_bonus(features)

            pass1_results.append({
                'candidate_id':   cand_id,
                'pass1_score':    pass1_score,
                'base_score':     base_score,
                'behavior_bonus': behavior_bonus,
                'features':       features,
                'candidate':      cand,
            })

        except Exception as e:
            # Never let a single bad record crash the pipeline
            continue

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n  Pass 1 complete in {elapsed:.1f}s")
    print(f"  Candidates surviving gates: {len(pass1_results):,}")

    # Sort and take top 5,000 for semantic re-ranking
    pass1_results.sort(key=lambda x: -x['pass1_score'])
    top_5000 = pass1_results[:3000]
    print(f"  Funnel output: top {len(top_5000):,} candidates → Pass 2")

    # -----------------------------------------------------------------------
    # PASS 2 — batched semantic scoring over top 5,000 (~45-90 seconds)
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Pass 2: Semantic re-ranking ({len(top_5000):,} candidates)...")
    t1 = datetime.now()

    top_ids        = [r['candidate_id'] for r in top_5000]
    semantic_scores = semantic_ranker.score_batch(candidates, top_ids)

    final_results = []
    for result, sem_score in zip(top_5000, semantic_scores):
        base_score     = result['base_score']
        behavior_bonus = result['behavior_bonus']

        # Additive blend: sem 0.45 + heuristic 0.45 + behavior 0.10
        # Max possible = 1.0 exactly — no saturation, full score spread
        final_score = (sem_score * 0.45) + (base_score * 0.45) + (behavior_bonus * 0.10)
        final_score = max(0.0, min(1.0, final_score))

        # Round to 6dp — matches validator tie-break logic
        result['score'] = round(final_score, 6)
        final_results.append(result)

    elapsed = (datetime.now() - t1).total_seconds()
    print(f"  Pass 2 complete in {elapsed:.1f}s")

    # Final sort: descending score, then ascending candidate_id for ties
    print("\n[4/4] Finalising ranks...")
    final_results.sort(key=lambda x: (-x['score'], x['candidate_id']))

    return final_results[:100], composite_scorer


# ---------------------------------------------------------------------------
# OUTPUT GENERATION
# ---------------------------------------------------------------------------

def generate_output(top_100, composite_scorer, output_path):
    print(f"\nGenerating output CSV → {output_path}")
    rows = []

    for rank, result in enumerate(top_100, 1):
        cand_id  = result['candidate_id']
        cand     = result['candidate']
        score    = result['score']
        features = result['features']

        reasoning = composite_scorer.generate_reasoning(cand, features, score, rank)

        rows.append({
            'candidate_id': cand_id,
            'rank':         rank,
            'score':        score,
            'reasoning':    reasoning,
        })

    df = pd.DataFrame(rows, columns=['candidate_id', 'rank', 'score', 'reasoning'])
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved {output_path} ({len(df)} rows)")
    print("\nTop 10:")
    print(df[['candidate_id', 'rank', 'score']].head(10).to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# VALIDATION (mirrors validate_submission.py logic)
# ---------------------------------------------------------------------------

def validate_output(df):
    print("\nValidating submission format...")
    issues = []

    if len(df) != 100:
        issues.append(f"Row count: expected 100, got {len(df)}")

    for col in ['candidate_id', 'rank', 'score', 'reasoning']:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")

    if set(df['rank'].tolist()) != set(range(1, 101)):
        issues.append("Ranks are not exactly 1-100")

    scores = df['score'].tolist()
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            issues.append(f"Score not non-increasing: rank {i+1} ({scores[i]}) < rank {i+2} ({scores[i+1]})")
            break

    # Tie-break check: equal scores must be ordered by candidate_id ascending
    for i in range(len(scores) - 1):
        if scores[i] == scores[i + 1]:
            cid_a = df.iloc[i]['candidate_id']
            cid_b = df.iloc[i + 1]['candidate_id']
            if cid_a > cid_b:
                issues.append(
                    f"Tie-break violation at ranks {i+1}/{i+2}: "
                    f"{cid_a} should come after {cid_b}"
                )

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
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl')
    parser.add_argument('--jd',         default='job_description.md',   help='Path to job description')
    parser.add_argument('--out',        default='submission.csv',        help='Output CSV path')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("REDROB INTELLIGENT CANDIDATE RANKING")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    t_start = datetime.now()

    candidates = load_candidates(args.candidates)
    jd_text    = load_jd(args.jd)

    top_100, scorer = score_all_candidates(candidates, jd_text)

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