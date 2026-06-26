# Redrob Hackathon v4 — Intelligent Candidate Ranking

**Team:** khikhi  
**Challenge:** Intelligent Candidate Discovery & Ranking  

---

## Approach

Two-pass heuristic-semantic pipeline:

- **Pass 1 (heuristic funnel, ~25s):** Scans all 100K candidates using O(1) string-blob matching. Applies hard disqualification gates (honeypots, ghost profiles, location, experience floor) and JD trap penalties. Outputs top 3,000 finalists.
- **Pass 2 (semantic re-ranking, ~60s):** Encodes all 3,000 finalists in a single vectorised batch using `all-MiniLM-L6-v2`. Final score is an additive blend: semantic (45%) + heuristic base (45%) + behavioral bonus (10%) across 9 Redrob signals.

Total runtime: ~180–230s on CPU.

---

## Repo structure

```
├── src/
│   ├── rank.py                  # Main pipeline runner
│   ├── feature_extractor.py     # Pass 1 heuristics
│   ├── composite_scorer.py      # Scoring + reasoning
│   └── semantic_ranker.py       # Pass 2 semantic re-ranker
├── data/
│   ├── sample_candidates.json   # 50-candidate sample for sandbox
│   └── job_description.md       # JD used for ranking
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Reproduce submission

```bash
python src/rank.py \
  --candidates ./candidates.jsonl.gz \
  --jd ./data/job_description.md \
  --out ./submission.csv
```

Also accepts uncompressed `.jsonl`:

```bash
python src/rank.py \
  --candidates ./candidates.jsonl \
  --jd ./data/job_description.md \
  --out ./submission.csv
```

Runs end-to-end in **≤5 minutes on CPU, 16GB RAM, no network required**.

---

## Sandbox

Google Colab notebook (runs on `sample_candidates.json`, ~50 candidates):  
**[INSERT COLAB LINK HERE]**

---

## Dependencies

- `sentence-transformers==2.7.0` — local embedding model (`all-MiniLM-L6-v2`)
- `numpy`, `pandas` — standard
- No external API calls during ranking