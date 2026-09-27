import sys
import pandas as pd
from collections import defaultdict

val_gt_file = 'data/processed/val_ground_truth.tsv'
cache_path = 'data/processed/val_cache_df_scores.parquet'
val_s1_file = 'data/processed/val_s1_sample.tsv'

s1_data = {}
with open(val_s1_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1_data[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

cand_data = {}
for src in ['train_source2.tsv', 'train_source3.tsv']:
    p = 'student_resource/dataset/train/' + src
    with open(p, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            cand_data[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '')

gt_dict = {}
with open(val_gt_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        gt_dict[p[0]] = set(m.strip() for m in p[1].split(',') if m.strip()) if len(p) > 1 and p[1].strip() else set()

df = pd.read_parquet(cache_path)
df['country'] = df['id1'].map(lambda x: s1_data.get(x, ('','',''))[2])

india_df = df[df['country'] == 'India'].copy()
india_gt = {k: v for k, v in gt_dict.items() if s1_data.get(k, ('','',''))[2] == 'India'}

# Check candidate recall in India
all_cands_india = set(zip(india_df['id1'], india_df['id2']))
gt_pairs_india = set()
for s1, ms in india_gt.items():
    for m in ms:
        gt_pairs_india.add((s1, m))

in_cands = len(gt_pairs_india.intersection(all_cands_india))
total_gt_in = len(gt_pairs_india)
print(f"India True GT Matches: {total_gt_in:,}")
print(f"Captured by Candidates: {in_cands:,} ({in_cands/total_gt_in*100:.2f}%)")
print(f"Missed by Candidates:   {total_gt_in - in_cands:,} ({(total_gt_in - in_cands)/total_gt_in*100:.2f}%)")

# Check model conversion in India
in_cands_df = india_df[india_df.apply(lambda r: (r['id1'], r['id2']) in gt_pairs_india, axis=1)]
print(f"\nModel Score Distribution for True Matches in India:")
print(in_cands_df['score'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
