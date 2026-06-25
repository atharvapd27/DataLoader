#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranker
Production pipeline - ranks 100K candidates for Senior AI Engineer role
"""

import json
import sys
import argparse
from datetime import datetime
import pandas as pd
from collections import defaultdict
import numpy as np

# Local imports
from feature_extractor_prod import FeatureExtractor
from semantic_ranker_prod import SemanticRanker
from composite_scorer_prod import CompositeScorer

def load_candidates(jsonl_path):
    """Load candidates from JSONL."""
    candidates = {}
    print(f"Loading candidates from {jsonl_path}...")
    
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            cand = json.loads(line)
            candidates[cand['candidate_id']] = cand
            
            if (i + 1) % 20000 == 0:
                print(f"  Loaded {i+1} candidates...")
    
    print(f"Total loaded: {len(candidates)}")
    return candidates

def load_jd(jd_path):
    """Load job description."""
    if jd_path.endswith('.docx'):
        from docx import Document
        doc = Document(jd_path)
        text = '\n'.join([p.text for p in doc.paragraphs])
    else:
        with open(jd_path, 'r') as f:
            text = f.read()
    
    return text

def score_all_candidates(candidates, jd_text, num_workers=1):
    """Score all candidates."""
    print("\n" + "="*60)
    print("SCORING PIPELINE")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing scorers...")
    feature_extractor = FeatureExtractor(candidates, jd_text)
    semantic_ranker = SemanticRanker(model_name='all-MiniLM-L6-v2')
    semantic_ranker.set_jd(jd_text)
    composite_scorer = CompositeScorer(semantic_ranker)
    
    # Score all candidates
    print("\n2. Extracting features and scoring...")
    results = []
    
    total = len(candidates)
    
    for idx, (cand_id, cand) in enumerate(candidates.items()):
        if (idx + 1) % 10000 == 0:
            pct = 100 * (idx + 1) / total
            print(f"  [{idx+1}/{total}] {pct:.1f}%")
        
        try:
            # Extract features
            features = feature_extractor.extract_all_features(cand_id)
            
            # Compute score
            score = composite_scorer.score_candidate(cand_id, cand, features)
            
            results.append({
                'candidate_id': cand_id,
                'score': score,
                'features': features,
                'candidate': cand  # Keep for reasoning
            })
        
        except Exception as e:
            print(f"  WARNING: Failed to score {cand_id}: {e}")
            continue
    
    print(f"\nSuccessfully scored {len(results)} candidates")
    
    # Sort by score
    print("\n3. Ranking candidates...")
    results.sort(key=lambda x: (-x['score'], x['candidate_id']))  # Descending score, then by ID
    
    return results[:100], composite_scorer

def generate_output(top_100, composite_scorer, output_path):
    """Generate final CSV output."""
    print(f"\n4. Generating output CSV...")
    
    rows = []
    
    for rank, result in enumerate(top_100, 1):
        cand_id = result['candidate_id']
        cand = result['candidate']
        score = result['score']
        features = result['features']
        
        # Generate reasoning
        reasoning = composite_scorer.generate_reasoning(
            cand, features, score, rank
        )
        
        rows.append({
            'candidate_id': cand_id,
            'rank': rank,
            'score': round(score, 6),
            'reasoning': reasoning
        })
    
    # Write CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved {output_path}")
    print(f"\nTop 10 candidates:")
    print(df[['candidate_id', 'rank', 'score']].head(10).to_string(index=False))
    
    return df

def validate_output(df):
    """Validate output format."""
    print("\n5. Validating submission format...")
    
    issues = []
    
    # Check row count
    if len(df) != 100:
        issues.append(f"Row count: expected 100, got {len(df)}")
    
    # Check columns
    required_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
    
    # Check ranks
    if set(df['rank']) != set(range(1, 101)):
        issues.append("Ranks not exactly 1-100")
    
    # Check scores non-increasing
    scores = df['score'].tolist()
    for i in range(len(scores) - 1):
        if scores[i] < scores[i+1]:
            issues.append(f"Scores not non-increasing (rank {i+1} to {i+2})")
            break
    
    # Check no empty reasoning
    empty_reasoning = df[df['reasoning'].isna() | (df['reasoning'] == '')].shape[0]
    if empty_reasoning > 0:
        issues.append(f"Empty reasoning in {empty_reasoning} rows")
    
    if issues:
        print("✗ Validation FAILED:")
        for issue in issues:
            print(f"    {issue}")
        return False
    else:
        print("✓ Validation PASSED")
        return True

def main():
    parser = argparse.ArgumentParser(description='Rank candidates for Redrob challenge')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl')
    parser.add_argument('--jd', default='job_description.docx', help='Path to job description')
    parser.add_argument('--out', default='submission.csv', help='Output CSV path')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("REDROB INTELLIGENT CANDIDATE RANKING")
    print("="*60)
    
    # Load data
    candidates = load_candidates(args.candidates)
    jd_text = load_jd(args.jd)
    
    # Score
    top_100, scorer = score_all_candidates(candidates, jd_text)
    
    # Generate output
    df = generate_output(top_100, scorer, args.out)
    
    # Validate
    is_valid = validate_output(df)
    
    if is_valid:
        print("\n" + "="*60)
        print("✓ READY FOR SUBMISSION")
        print("="*60)
        sys.exit(0)
    else:
        print("\n✗ Fix validation errors before submission")
        sys.exit(1)

if __name__ == '__main__':
    main()