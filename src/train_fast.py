#!/usr/bin/env python3
"""
Fast, robust training script for Stage 1 LightGBM matching model.
Creates high-quality training pairs (True Positives from ground truth + Hard Negatives from blocking)
and trains a calibrated LightGBM model saved to output/stage1_lgb_model.txt.
"""

import os
import sys
import time
import random
import re
import unicodedata
import pandas as pd
import numpy as np
import lightgbm as lgb
import jellyfish
import rapidfuzz

from matching_pipeline import FEATURE_COLUMNS, compute_features_fast

def main():
    start_time = time.time()
    print("=" * 80)
    print(" TRAINING STAGE 1 LIGHTGBM MODEL ON GROUND TRUTH + HARD NEGATIVES")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'train')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, 'stage1_lgb_model.txt')

    # 1. Load Ground Truth
    print("\n[1/5] Loading Ground Truth from train_ground_truth.tsv...")
    gt = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t', dtype=str)
    gt_clean = gt.dropna(subset=['matched_entity_ids']).copy()
    gt_clean['matched_entity_ids'] = gt_clean['matched_entity_ids'].astype(str).str.split(',')
    gt_flat = gt_clean.explode('matched_entity_ids')
    gt_flat['matched_entity_ids'] = gt_flat['matched_entity_ids'].str.strip()

    true_pairs = list(zip(gt_flat['source1_entity_id'], gt_flat['matched_entity_ids']))
    true_pairs_set = set(true_pairs)
    print(f"   -> Total Ground Truth True Positive pairs: {len(true_pairs):,}")

    # Sample 150,000 true positive pairs for fast, balanced training
    random.seed(42)
    sample_pos_pairs = random.sample(true_pairs, min(150000, len(true_pairs)))
    print(f"   -> Sampled {len(sample_pos_pairs):,} positive pairs for training.")

    needed_s1 = set(p[0] for p in sample_pos_pairs)
    needed_cand = set(p[1] for p in sample_pos_pairs)

    # 2. Load Source Dictionaries for Sampled Entities + Hard Negatives
    print("\n[2/5] Loading entity attributes from train sources...")
    t0 = time.time()
    
    # Load Source 1
    s1_dict = {}
    with open(os.path.join(train_dir, 'train_source1.tsv'), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            eid = p[0]
            if eid in needed_s1 or len(s1_dict) < 200000:
                s1_dict[eid] = (
                    p[1] if len(p) > 1 and p[1] else '',
                    p[2] if len(p) > 2 and p[2] else '',
                    p[3] if len(p) > 3 and p[3] else ''
                )
    print(f"   -> Loaded {len(s1_dict):,} S1 entities.")

    # Load Source 2 & 3
    cand_dict = {}
    tokens_to_cand = {} # token -> list of cand IDs for name-sharing hard negatives
    zip_to_cand = {}    # zip/pin -> list of cand IDs for address-sharing hard negatives
    for src in ['train_source2.tsv', 'train_source3.tsv']:
        c_count = 0
        with open(os.path.join(train_dir, src), 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                eid = p[0]
                nm = p[1] if len(p) > 1 and p[1] else ''
                ad = p[2] if len(p) > 2 and p[2] else ''
                ct = p[3] if len(p) > 3 and p[3] else ''
                if eid in needed_cand or (len(cand_dict) < 300000 and c_count < 150000):
                    cand_dict[eid] = (nm, ad, ct)
                    c_count += 1
                    toks = [t for t in nm.lower().split() if len(t) >= 4]
                    if toks:
                        first_tok = toks[0]
                        if first_tok not in tokens_to_cand:
                            tokens_to_cand[first_tok] = []
                        if len(tokens_to_cand[first_tok]) < 20:
                            tokens_to_cand[first_tok].append(eid)
                    z_m = re.search(r'\b\d{5,6}\b', ad)
                    if z_m:
                        zp = z_m.group(0)
                        if zp not in zip_to_cand:
                            zip_to_cand[zp] = []
                        if len(zip_to_cand[zp]) < 20:
                            zip_to_cand[zp].append(eid)
        print(f"   -> Loaded {c_count:,} entities from {src}.")
    print(f"   -> Total candidate entities in lookup: {len(cand_dict):,} in {time.time()-t0:.2f}s.")

    # 3. Generate Hard Negatives (Both Name-sharing AND Address-sharing)
    print("\n[3/5] Constructing balanced training dataset with Hard Negatives...")
    all_s1 = []
    all_cand = []
    all_targets = []

    # Add Positives
    for s1, c2 in sample_pos_pairs:
        if s1 in s1_dict and c2 in cand_dict:
            all_s1.append(s1)
            all_cand.append(c2)
            all_targets.append(1)

    n_pos = len(all_targets)
    print(f"   -> Valid Positive training pairs: {n_pos:,}")

    # Add Hard Negatives (1x Name-sharing, 1.5x Address-sharing)
    neg_name_count = 0
    neg_addr_count = 0
    s1_keys = list(s1_dict.keys())
    random.shuffle(s1_keys)

    for s1 in s1_keys:
        nm, ad, ct = s1_dict[s1]
        
        # Name-collision hard negatives
        if neg_name_count < n_pos:
            toks = [t for t in nm.lower().split() if len(t) >= 4]
            if toks:
                first_tok = toks[0]
                for c2 in tokens_to_cand.get(first_tok, []):
                    if (s1, c2) not in true_pairs_set and c2 in cand_dict:
                        if ct and cand_dict[c2][2] and ct == cand_dict[c2][2]:
                            all_s1.append(s1)
                            all_cand.append(c2)
                            all_targets.append(0)
                            neg_name_count += 1
                            if neg_name_count >= n_pos:
                                break

        # Address-collision hard negatives (Crucial for eliminating same-street/same-building false positives)
        if neg_addr_count < int(n_pos * 1.5):
            z_m = re.search(r'\b\d{5,6}\b', ad)
            if z_m:
                zp = z_m.group(0)
                for c2 in zip_to_cand.get(zp, []):
                    if (s1, c2) not in true_pairs_set and c2 in cand_dict:
                        if ct and cand_dict[c2][2] and ct == cand_dict[c2][2]:
                            all_s1.append(s1)
                            all_cand.append(c2)
                            all_targets.append(0)
                            neg_addr_count += 1
                            if neg_addr_count >= int(n_pos * 1.5):
                                break

    print(f"   -> Generated {neg_name_count:,} name-sharing and {neg_addr_count:,} address-sharing hard negatives.")
    print(f"   -> Total training pairs: {len(all_targets):,} ({n_pos:,} Positives, {neg_name_count + neg_addr_count:,} Hard Negatives).")

    # 4. Feature Computation
    print("\n[4/5] Computing 16 Vectorized Similarity & Disqualifier Features...")
    t0 = time.time()
    df_train = pd.DataFrame({
        'id1': all_s1,
        'id2': all_cand,
        'name1': [s1_dict[x][0] for x in all_s1],
        'addr1': [s1_dict[x][1] for x in all_s1],
        'country1': [s1_dict[x][2] for x in all_s1],
        'name2': [cand_dict[x][0] for x in all_cand],
        'addr2': [cand_dict[x][1] for x in all_cand],
        'country2': [cand_dict[x][2] for x in all_cand],
        'target': all_targets
    })

    df_train = compute_features_fast(df_train)
    print(f"   -> Features computed in {time.time()-t0:.2f}s.")

    # 5. Train LightGBM Model with Precision Bias (scale_pos_weight=0.25)
    print("\n[5/5] Training LightGBM Booster with precision penalty (scale_pos_weight=0.25)...")
    features = FEATURE_COLUMNS
    X = df_train[features]
    y = df_train['target']

    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'scale_pos_weight': 0.25, # Heavily penalize false positives for F0.5
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(params, train_data, num_boost_round=120)
    model.save_model(model_save_path)
    print(f"\n SUCCESS: Stage 1 LightGBM model saved to {model_save_path}")

    # Print Feature Importances
    print("\n Feature Importances (Gain):")
    importances = sorted(zip(features, model.feature_importance(importance_type='gain')), key=lambda x: x[1], reverse=True)
    for feat, imp in importances:
        print(f"   - {feat:30s}: {imp:10.2f}")

    print("=" * 80)
    print(f" TRAINING FINISHED IN {(time.time()-start_time)/60:.2f} MINUTES!")
    print("=" * 80)

if __name__ == '__main__':
    main()
