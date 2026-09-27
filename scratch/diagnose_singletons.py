import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import lightgbm as lgb

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
val_s1_file = os.path.join(base_dir, 'data', 'processed', 'val_s1_sample.tsv')
val_cand_file = os.path.join(base_dir, 'data', 'processed', 'candidate_pairs_val.tsv')
val_gt_file = os.path.join(base_dir, 'data', 'processed', 'val_ground_truth.tsv')
model_file = os.path.join(base_dir, 'output', 'stage1_lgb_model.txt')
train_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'train')

gt_dict = {}
with open(val_gt_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1 = p[0]
        ms = set(m.strip() for m in p[1].split(',') if m.strip()) if len(p) > 1 and p[1].strip() else set()
        gt_dict[s1] = ms

true_singletons = set(s1 for s1, ms in gt_dict.items() if len(ms) == 0)

cands_for_singletons = []
with open(val_cand_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1 = p[0]
        if s1 in true_singletons and len(p) > 1 and p[1].strip():
            cs = [c.strip() for c in p[1].split(',') if c.strip()]
            for c in cs:
                cands_for_singletons.append((s1, c))
                if len(cands_for_singletons) >= 1500:
                    break
        if len(cands_for_singletons) >= 1500:
            break

s1_ids = set(p[0] for p in cands_for_singletons)
cand_ids = set(p[1] for p in cands_for_singletons)

s1_data = {}
with open(val_s1_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_ids:
            s1_data[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

cand_data = {}
for src in ['train_source2.tsv', 'train_source3.tsv']:
    with open(os.path.join(train_dir, src), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in cand_ids:
                cand_data[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

sys.path.insert(0, os.path.join(base_dir, 'src'))
from matching_pipeline import compute_features_fast, FEATURE_COLUMNS

df_test = pd.DataFrame({
    'id1': [p[0] for p in cands_for_singletons if p[0] in s1_data and p[1] in cand_data],
    'id2': [p[1] for p in cands_for_singletons if p[0] in s1_data and p[1] in cand_data],
    'name1': [s1_data[p[0]][0] for p in cands_for_singletons if p[0] in s1_data and p[1] in cand_data],
    'addr1': [s1_data[p[0]][1] for p in cands_for_singletons if p[0] in s1_data and p[1] in cand_data],
    'country1': [s1_data[p[0]][2] for p in cands_for_singletons if p[0] in s1_data and p[1] in cand_data],
    'name2': [cand_data[p[1]][0] for p in cands_for_singletons if p[0] in s1_data and p[1] in cand_data],
    'addr2': [cand_data[p[1]][1] for p in cands_for_singletons if p[0] in s1_data and p[1] in cand_data],
    'country2': [cand_data[p[1]][2] for p in cands_for_singletons if p[0] in s1_data and p[1] in cand_data],
})

df_feats = compute_features_fast(df_test)
model = lgb.Booster(model_file=model_file)
df_feats['score'] = model.predict(df_feats[FEATURE_COLUMNS])

fp_pairs = df_feats[df_feats['score'] >= 0.35]
print('Pairs involving singletons with score >= 0.35:', len(fp_pairs))
for _, r in fp_pairs.head(15).iterrows():
    print(f"Score: {r['score']:.4f} | S1: [{r['name1']}] [{r['addr1']}] vs C2: [{r['name2']}] [{r['addr2']}]")
