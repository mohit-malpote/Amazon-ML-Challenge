import sys, os, time, gc
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
from matching_pipeline import (
    clean_brand_name, clean_legal, get_first_token, get_leading_num, get_zip,
    SemanticInspector, FEATURE_COLUMNS
)
from data_processing import format_predictions_for_submission
import jellyfish, rapidfuzz

print("Testing End-to-End Pipeline on first 1,000 S1 entities...")

# 1. Load test S1 first 1000
test_s1_dict = {}
with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for _ in range(1000):
        line = f.readline()
        if not line: break
        p = line.rstrip('\n').split('\t')
        eid = p[0]
        nm = p[1].lower() if len(p) > 1 else ''
        ad = p[2].lower() if len(p) > 2 else ''
        ct = p[3] if len(p) > 3 else ''
        test_s1_dict[eid] = {
            'name': nm, 'addr': ad, 'country': ct,
            'token': get_first_token(nm), 'legal': clean_legal(nm),
            'num': get_leading_num(ad), 'zip': get_zip(ad)
        }

print(f"Loaded {len(test_s1_dict)} S1 entities.")

# Read candidate pairs for first 1000 S1 entities
cand_ids_needed = set()
s1_cand_pairs = []
with open('output/candidate_pairs_test.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for _ in range(1000):
        line = f.readline()
        if not line: break
        s1, _, cands = line.rstrip('\n').partition('\t')
        if not cands: continue
        cand_list = cands.split(',')
        cand_ids_needed.update(cand_list)
        s1_cand_pairs.append((s1, cand_list))

print(f"Found {sum(len(c) for _, c in s1_cand_pairs):,} candidate pairs referencing {len(cand_ids_needed):,} unique S2/S3 IDs.")

# Load only the needed candidate entities from test_source2 and test_source3
cand_dict = {}
for src in ['test_source2.tsv', 'test_source3.tsv']:
    with open(f'student_resource/dataset/test/{src}', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            eid = p[0]
            if eid in cand_ids_needed:
                nm = p[1].lower() if len(p) > 1 else ''
                ad = p[2].lower() if len(p) > 2 else ''
                ct = p[3] if len(p) > 3 else ''
                cand_dict[eid] = {
                    'name': nm, 'addr': ad, 'country': ct,
                    'token': get_first_token(nm), 'legal': clean_legal(nm),
                    'num': get_leading_num(ad), 'zip': get_zip(ad)
                }

print(f"Loaded {len(cand_dict):,} candidate entities into memory.")

# Build pairs and filter cross-country
valid_pairs = []
for s1, c_list in s1_cand_pairs:
    e1 = test_s1_dict[s1]
    c1 = e1['country']
    for c2 in c_list:
        e2 = cand_dict.get(c2)
        if not e2: continue
        c2_ct = e2['country']
        if c1 and c2_ct and c1 != c2_ct:
            continue
        valid_pairs.append((s1, c2))

print(f"Valid pairs after country filter: {len(valid_pairs):,}")

# Feature extraction
n1 = [test_s1_dict[p[0]]['name'] for p in valid_pairs]
n2 = [cand_dict[p[1]]['name'] for p in valid_pairs]
a1 = [test_s1_dict[p[0]]['addr'] for p in valid_pairs]
a2 = [cand_dict[p[1]]['addr'] for p in valid_pairs]
f1 = [test_s1_dict[p[0]]['token'] for p in valid_pairs]
f2 = [cand_dict[p[1]]['token'] for p in valid_pairs]
cl1 = [test_s1_dict[p[0]]['legal'] for p in valid_pairs]
cl2 = [cand_dict[p[1]]['legal'] for p in valid_pairs]
num1 = [test_s1_dict[p[0]]['num'] for p in valid_pairs]
num2 = [cand_dict[p[1]]['num'] for p in valid_pairs]
z1 = [test_s1_dict[p[0]]['zip'] for p in valid_pairs]
z2 = [cand_dict[p[1]]['zip'] for p in valid_pairs]

feat_dict = {
    'name_levenshtein': [jellyfish.levenshtein_distance(a, b) for a, b in zip(n1, n2)],
    'name_jaro_winkler': [jellyfish.jaro_winkler_similarity(a, b) for a, b in zip(n1, n2)],
    'name_token_set_ratio': [rapidfuzz.fuzz.token_set_ratio(a, b) for a, b in zip(n1, n2)],
    'address_street_match': [rapidfuzz.fuzz.token_set_ratio(a, b) for a, b in zip(a1, a2)],
    'address_zip_match': [1 if (x and y and x == y) else (0 if (x and y and x != y) else -1) for x, y in zip(z1, z2)],
    'postal_code_mismatch': [1 if (x and y and x != y) else 0 for x, y in zip(z1, z2)],
    'street_number_mismatch': [1 if (x and y and x != y) else 0 for x, y in zip(num1, num2)],
    'street_number_match': [1 if (x and y and x == y) else 0 for x, y in zip(num1, num2)],
    'brand_first_token_ratio': [rapidfuzz.fuzz.ratio(x, y) / 100.0 if (x and y) else 0.5 for x, y in zip(f1, f2)],
    'clean_legal_name_match': [1 if (x and x == y) else 0 for x, y in zip(cl1, cl2)],
    'name_length_ratio': [min(len(a), len(b)) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0.0 for a, b in zip(n1, n2)],
    'address_length_ratio': [min(len(a), len(b)) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0.0 for a, b in zip(a1, a2)],
    'name_to_address_cross_match': [max(1 if (x and x in y) else 0, 1 if (y and y in x) else 0) for x, y in zip(n1, a2)],
    'address_name_ratio_diff': [a - n for a, n in zip([rapidfuzz.fuzz.token_set_ratio(a, b) for a, b in zip(a1, a2)], [rapidfuzz.fuzz.token_set_ratio(a, b) for a, b in zip(n1, n2)])],
    'exact_name_match': [1 if (a and a == b) else 0 for a, b in zip(n1, n2)],
    'country_mismatch': [0] * len(valid_pairs)
}

X = pd.DataFrame(feat_dict)[FEATURE_COLUMNS]

# LightGBM inference
model = lgb.Booster(model_file='output/stage1_lgb_model.txt')
s1_scores = model.predict(X)

# Decision logic
st_mismatch = np.array(feat_dict['street_number_mismatch'])
tsr = np.array(feat_dict['name_token_set_ratio'])
is_building_conflict = (st_mismatch == 1) & (tsr < 75)

matches = []
borderline_indices = []

for idx, (p, score, conflict) in enumerate(zip(valid_pairs, s1_scores, is_building_conflict)):
    if conflict or score <= 0.20:
        continue
    elif score >= 0.90:
        matches.append((p[0], p[1], float(score)))
    else:
        borderline_indices.append(idx)

print(f"Stage 1 direct matches: {len(matches):,} | Borderline to Stage 2: {len(borderline_indices):,}")

# Stage 2 on GPU
if borderline_indices:
    inspector = SemanticInspector()
    pairs_s2 = [(n1[i], a1[i], n2[i], a2[i]) for i in borderline_indices]
    s2_scores = inspector.predict(pairs_s2)
    b_tokens = [feat_dict['brand_first_token_ratio'][i] for i in borderline_indices]
    
    s2_accepted = 0
    for idx, s2_s, bt in zip(borderline_indices, s2_scores, b_tokens):
        if s2_s >= 0.88 and bt >= 0.40:
            p = valid_pairs[idx]
            matches.append((p[0], p[1], float(s2_s)))
            s2_accepted += 1
    print(f"Stage 2 accepted {s2_accepted:,} borderline matches.")

matches_df = pd.DataFrame(matches, columns=['id1', 'id2', 'score'])
print(f"Total raw matches before injective deduplication: {len(matches_df):,}")

# Injective deduplication
matches_df = matches_df.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
print(f"Matches after injective deduplication: {len(matches_df):,}")

# Format for 1000 S1 IDs
s1_sample_ids = list(test_s1_dict.keys())
sub_sample = format_predictions_for_submission(matches_df, s1_sample_ids)
print(f"Formatted submission shape: {sub_sample.shape}")
print("Sample output:")
print(sub_sample.head(10))
