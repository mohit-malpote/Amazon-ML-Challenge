import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
import os
import gc
import time
from matching_pipeline import compute_features_fast, SemanticInspector, SupremeJudge, run_cascade_pipeline, FEATURE_COLUMNS
import data_processing

def calculate_f05_score(true_matches, pred_matches):
    """
    Macro-averaged F0.5 score per S1 entity as specified in ML Challenge 2026.
    true_matches: dict mapping s1_id -> list of true match ids
    pred_matches: dict mapping s1_id -> list of predicted match ids
    """
    precisions = []
    recalls = []
    
    for s1_id, true_list in true_matches.items():
        pred_list = pred_matches.get(s1_id, [])
        
        # Singleton logic
        if not true_list and not pred_list:
            precisions.append(1.0)
            recalls.append(1.0)
            continue
        elif not true_list and pred_list:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        elif true_list and not pred_list:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        true_set = set(true_list)
        pred_set = set(pred_list)
        
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        
    p = float(np.mean(precisions))
    r = float(np.mean(recalls))
    f05 = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0.0
    return f05, p, r

def calculate_micro_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0.0
    return f05, p, r, tp, fp, fn

def main():
    print("=" * 70)
    print(" ML CHALLENGE 2026: STAGE 1 + STAGE 2 MATCHING PIPELINE")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'train')
    output_dir = os.path.join(base_dir, 'output')
    
    s1_path = os.path.join(dataset_dir, 'train_source1.tsv')
    s2_path = os.path.join(dataset_dir, 'train_source2.tsv')
    s3_path = os.path.join(dataset_dir, 'train_source3.tsv')
    gt_path = os.path.join(dataset_dir, 'train_ground_truth.tsv')
    
    candidates_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    if not os.path.exists(candidates_path):
        candidates_path = os.path.join(output_dir, 'candidate_pairs.parquet')
        
    print(f"\n1. Loading Ground Truth from {gt_path}...")
    gt_raw = pd.read_csv(gt_path, sep='\t', dtype=str)
    gt_clean = gt_raw.dropna(subset=['matched_entity_ids']).copy()
    gt_clean['matched_entity_ids'] = gt_clean['matched_entity_ids'].astype(str).str.split(',')
    gt_flat = gt_clean.explode('matched_entity_ids')
    true_pairs = set(zip(gt_flat['source1_entity_id'], gt_flat['matched_entity_ids'].str.strip()))
    print(f"   Loaded {len(true_pairs):,} ground truth positive pairs.")
    del gt_raw, gt_clean, gt_flat; gc.collect()
    
    print(f"\n2. Loading Candidate Pairs from {candidates_path}...")
    t0 = time.time()
    # Read first 400,000 candidate pairs using PyArrow for speed and low memory
    table = pq.read_table(candidates_path, columns=['source1_entity_id', 'candidate_entity_id'])
    SAMPLE_SIZE = min(400000, table.num_rows)
    df_cands = table.slice(0, SAMPLE_SIZE).to_pandas()
    df_cands = df_cands.rename(columns={'source1_entity_id': 'id1', 'candidate_entity_id': 'id2'})
    print(f"   Loaded sample of {len(df_cands):,} candidate pairs in {time.time()-t0:.2f}s.")
    del table; gc.collect()
    
    print(f"\n3. Loading Source Data Dictionaries for name & address lookup...")
    entity_dict = data_processing.load_source_data(s1_path, s2_path, s3_path)
    
    print("   Mapping business names, addresses, countries, and ground truth labels...")
    df_cands['name1'] = df_cands['id1'].map(lambda x: entity_dict.get(x, {}).get('business_name', ''))
    df_cands['addr1'] = df_cands['id1'].map(lambda x: entity_dict.get(x, {}).get('business_address', ''))
    df_cands['country1'] = df_cands['id1'].map(lambda x: entity_dict.get(x, {}).get('country', ''))
    
    df_cands['name2'] = df_cands['id2'].map(lambda x: entity_dict.get(x, {}).get('business_name', ''))
    df_cands['addr2'] = df_cands['id2'].map(lambda x: entity_dict.get(x, {}).get('business_address', ''))
    df_cands['country2'] = df_cands['id2'].map(lambda x: entity_dict.get(x, {}).get('country', ''))
    
    cand_tuples = list(zip(df_cands['id1'], df_cands['id2']))
    df_cands['target'] = [1 if p in true_pairs else 0 for p in cand_tuples]
    num_pos = df_cands['target'].sum()
    print(f"   Sample distribution: {num_pos:,} True Matches ({num_pos/len(df_cands)*100:.2f}%) | {len(df_cands)-num_pos:,} Hard Negatives.")
    
    # 4. Grouped Train/Validation Split by S1 ID
    print(f"\n4. Splitting into Train (80%) and Validation (20%) by unique S1 Entity ID...")
    s1_ids = list(set(df_cands['id1']))
    train_ids, val_ids = train_test_split(s1_ids, test_size=0.2, random_state=42)
    val_id_set = set(val_ids)
    
    df_train = df_cands[~df_cands['id1'].isin(val_id_set)].copy().reset_index(drop=True)
    df_val = df_cands[df_cands['id1'].isin(val_id_set)].copy().reset_index(drop=True)
    del df_cands; gc.collect()
    
    print(f"   Train Set: {len(df_train):,} pairs ({df_train['id1'].nunique():,} unique S1 entities)")
    print(f"   Val Set:   {len(df_val):,} pairs ({df_val['id1'].nunique():,} unique S1 entities)")
    
    # 5. Feature Engineering
    print(f"\n5. Computing Stage 1 Vectorized Features (Rapidfuzz + Jellyfish + Disqualifiers)...")
    t0 = time.time()
    df_train = compute_features_fast(df_train)
    df_val = compute_features_fast(df_val)
    print(f"   Features computed for both train and validation in {time.time()-t0:.2f}s.")
    
    features = FEATURE_COLUMNS
    print(f"   Using {len(features)} engineered features: {features}")
    
    # 6. Training LightGBM (Stage 1)
    print(f"\n6. Training LightGBM (Stage 1 GBDT with precision-biased scale_pos_weight=0.3)...")
    lgb_train = lgb.Dataset(df_train[features], df_train['target'])
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'scale_pos_weight': 0.3, # Precision bias for F0.5
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    model = lgb.train(params, lgb_train, num_boost_round=80)
    
    # Save model artifact
    model_save_path = os.path.join(output_dir, 'stage1_lgb_model.txt')
    model.save_model(model_save_path)
    print(f"   Saved Stage 1 LightGBM model to {model_save_path}.")
    
    # Feature Importances
    print("\n   Stage 1 Feature Importances:")
    importances = sorted(zip(features, model.feature_importance(importance_type='gain')), key=lambda x: x[1], reverse=True)
    for feat, imp in importances:
        print(f"     - {feat:30s}: {imp:10.2f}")
        
    # 7. Evaluating Stage 1 Alone with Threshold Grid Search
    print(f"\n7. Evaluating Stage 1 Baseline on Validation Set...")
    s1_val_preds = model.predict(df_val[features])
    df_val['stage1_score'] = s1_val_preds
    
    val_true_dict = df_val[df_val['target'] == 1].groupby('id1')['id2'].apply(list).to_dict()
    for s1 in val_ids:
        if s1 not in val_true_dict: val_true_dict[s1] = []
        
    best_f05, best_t, best_p, best_r = 0.0, 0.85, 0.0, 0.0
    print("   Searching optimal standalone Stage 1 threshold:")
    for t in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
        cand_matches = df_val[df_val['stage1_score'] >= t]
        pred_dict = cand_matches.groupby('id1')['id2'].apply(list).to_dict()
        for s1 in val_ids:
            if s1 not in pred_dict: pred_dict[s1] = []
        f05_t, p_t, r_t = calculate_f05_score(val_true_dict, pred_dict)
        print(f"     Threshold {t:.2f} -> F0.5: {f05_t:.4f} | Precision: {p_t:.4f} | Recall: {r_t:.4f}")
        if f05_t > best_f05:
            best_f05, best_t, best_p, best_r = f05_t, t, p_t, r_t
            
    f05_s1, p_s1, r_s1 = best_f05, best_p, best_r
    print(f"   -> Optimal Standalone Stage 1 (Threshold {best_t:.2f}): Macro F0.5: {f05_s1:.4f} | Precision: {p_s1:.4f} | Recall: {r_s1:.4f}")
    
    # 8. Executing Stage 2 (Semantic Inspector Cross-Encoder)
    print(f"\n8. Initializing Stage 2 (Semantic Inspector: cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    semantic_inspector = SemanticInspector()
    
    # Calibrated Cascade Partitioning
    CLEAR_MATCH_THRESH = 0.90
    CLEAR_REJECT_THRESH = 0.20
    
    df_val['decision'] = 'PENDING'
    df_val.loc[df_val['stage1_score'] >= CLEAR_MATCH_THRESH, 'decision'] = 'MATCH'
    df_val.loc[df_val['stage1_score'] <= CLEAR_REJECT_THRESH, 'decision'] = 'REJECT'
    
    # Immediate Hard Disqualifications
    if 'country_mismatch' in df_val.columns:
        n_cm = (df_val['country_mismatch'] == 1).sum()
        df_val.loc[df_val['country_mismatch'] == 1, 'decision'] = 'REJECT'
        print(f"   Applied Country Disqualification: Filtered {n_cm:,} cross-country candidates.")
        
    st_and_name_mismatch = (df_val['street_number_mismatch'] == 1) & (df_val['name_token_set_ratio'] < 80)
    n_snm = st_and_name_mismatch.sum()
    df_val.loc[st_and_name_mismatch, 'decision'] = 'REJECT'
    print(f"   Applied Street Number Mismatch Disqualification: Filtered {n_snm:,} conflicting building pairs.")
    
    n_clear_match = (df_val['decision'] == 'MATCH').sum()
    n_clear_reject = (df_val['decision'] == 'REJECT').sum()
    n_borderline = (df_val['decision'] == 'PENDING').sum()
    
    print(f"\n   Validation Set Cascade Partitioning:")
    print(f"     - Clear Matches directly accepted by Stage 1: {n_clear_match:,} ({n_clear_match/len(df_val)*100:.1f}%)")
    print(f"     - Clear Rejects directly filtered by Stage 1:  {n_clear_reject:,} ({n_clear_reject/len(df_val)*100:.1f}%)")
    print(f"     - Borderline Pairs escalated to Stage 2:       {n_borderline:,} ({n_borderline/len(df_val)*100:.1f}%)")
    
    stage2_mask = df_val['decision'] == 'PENDING'
    stage2_df = df_val[stage2_mask].copy()
    
    if len(stage2_df) > 0:
        print(f"\n   Running Stage 2 Cross-Encoder Inference on {len(stage2_df)} borderline pairs...")
        t_s2 = time.time()
        pairs_for_s2 = list(zip(stage2_df['name1'], stage2_df['addr1'], stage2_df['name2'], stage2_df['addr2']))
        s2_scores = semantic_inspector.predict(pairs_for_s2)
        df_val.loc[stage2_mask, 'stage2_score'] = s2_scores
        print(f"   Stage 2 Inference completed in {time.time()-t_s2:.2f}s ({len(stage2_df)/(time.time()-t_s2):.1f} pairs/sec).")
        
        # Threshold grid search for Stage 2 on borderline set
        print("   Evaluating Stage 2 Thresholds on Borderline Pairs:")
        best_s2_t, best_casc_f05 = 0.88, 0.0
        for s2_t in [0.50, 0.70, 0.80, 0.85, 0.88, 0.90]:
            temp_dec = df_val['decision'].copy()
            temp_dec.loc[stage2_mask & (df_val['stage2_score'] >= s2_t)] = 'MATCH'
            temp_dec.loc[stage2_mask & (df_val['stage2_score'] < s2_t)] = 'REJECT'
            temp_matches = df_val[temp_dec == 'MATCH']
            p_dict = temp_matches.groupby('id1')['id2'].apply(list).to_dict()
            for s1 in val_ids:
                if s1 not in p_dict: p_dict[s1] = []
            c_f05, c_p, c_r = calculate_f05_score(val_true_dict, p_dict)
            print(f"     S2 Threshold {s2_t:.2f} -> Cascade F0.5: {c_f05:.4f} | Precision: {c_p:.4f} | Recall: {c_r:.4f}")
            if c_f05 > best_casc_f05:
                best_casc_f05 = c_f05
                best_s2_t = s2_t
                
        print(f"   -> Calibrated Optimal Stage 2 Threshold: {best_s2_t:.2f}")
        df_val.loc[stage2_mask & (df_val['stage2_score'] >= best_s2_t), 'decision'] = 'MATCH'
        df_val.loc[stage2_mask & (df_val['stage2_score'] < best_s2_t), 'decision'] = 'REJECT'
        
        # Evaluate Stage 2 accuracy on borderline cases specifically
        s2_subset = df_val[stage2_mask]
        s2_preds = (s2_subset['decision'] == 'MATCH').astype(int)
        s2_f05, s2_p, s2_r, s2_tp, s2_fp, s2_fn = calculate_micro_metrics(s2_subset['target'], s2_preds)
        print(f"\n   Stage 2 Performance on Borderline Subset Alone:")
        print(f"     - Borderline True Matches Recovered:  {s2_tp:,}")
        print(f"     - Borderline False Matches Prevented: {len(s2_subset) - (s2_tp + s2_fp):,}")
        print(f"     - Borderline Precision:               {s2_p:.4f}")
        print(f"     - Borderline Recall:                  {s2_r:.4f}")
        print(f"     - Borderline F0.5:                    {s2_f05:.4f}")
    
    # 9. Overall Stage 1 + Stage 2 Cascade Evaluation on Validation Set
    print(f"\n9. Stage 1 + Stage 2 Cascade Evaluation on Validation Set...")
    cascade_matches = df_val[df_val['decision'] == 'MATCH']
    cascade_pred_dict = cascade_matches.groupby('id1')['id2'].apply(list).to_dict()
    for s1 in val_ids:
        if s1 not in cascade_pred_dict: cascade_pred_dict[s1] = []
        
    f05_casc, p_casc, r_casc = calculate_f05_score(val_true_dict, cascade_pred_dict)
    
    # 10. Initializing & Executing Stage 3 (Supreme Judge LLM Precision Auditor)
    print(f"\n10. Initializing Stage 3 (Supreme Judge Veto Auditor)...")
    supreme_judge = SupremeJudge()
    
    df_val['decision_s3'] = df_val['decision'].copy() # Initialize with 2-stage decisions
    
    # Target Population for Stage 3 Veto Auditor:
    # 1. Pairs where S2 was uncertain: 0.45 <= S2 < best_s2_t
    # 2. Or pairs where S2 accepted (>= best_s2_t), BUT brand first token is drastically different (< 0.40)
    s3_mask = stage2_mask & (
        ((df_val['stage2_score'] >= 0.45) & (df_val['stage2_score'] < best_s2_t)) |
        ((df_val['stage2_score'] >= best_s2_t) & (df_val['brand_first_token_ratio'] < 0.40) & (df_val['name_token_set_ratio'] < 75))
    )
    s3_df = df_val[s3_mask].copy()
    print(f"   Stage 3 Suspect Pairs for Precision Audit: {len(s3_df):,} pairs ({len(s3_df)/len(df_val)*100:.2f}%)")
    
    if len(s3_df) > 0:
        max_audit = 60 if supreme_judge.device.type == "cpu" else len(s3_df)
        s3_sample = s3_df.head(max_audit)
        print(f"   Executing Stage 3 Supreme Judge Veto Audit on {len(s3_sample)} suspect pairs (device: {supreme_judge.device})...")
        t_s3 = time.time()
        s3_pairs = list(zip(s3_sample['name1'], s3_sample['addr1'], s3_sample['name2'], s3_sample['addr2']))
        s3_decisions = supreme_judge.judge_pairs(s3_pairs, batch_size=16)
        df_val.loc[s3_sample.index, 'decision_s3'] = s3_decisions
        print(f"   Stage 3 Arbitration completed in {time.time()-t_s3:.2f}s ({len(s3_sample)/(time.time()-t_s3):.1f} pairs/sec).")
        
        # Micro metrics on the audited subset alone
        s3_preds = (df_val.loc[s3_sample.index, 'decision_s3'] == 'MATCH').astype(int)
        s3_f05, s3_p, s3_r, s3_tp, s3_fp, s3_fn = calculate_micro_metrics(df_val.loc[s3_sample.index, 'target'], s3_preds)
        print(f"\n   Stage 3 Performance on Audited Subset Alone:")
        print(f"     - Ambiguous True Matches Recovered:  {s3_tp:,}")
        print(f"     - Ambiguous Precision:               {s3_p:.4f}")
        print(f"     - Ambiguous Recall:                  {s3_r:.4f}")
        print(f"     - Ambiguous F0.5:                    {s3_f05:.4f}")
        
    # Evaluate Complete 3-Stage Cascade
    s3_matches = df_val[df_val['decision_s3'] == 'MATCH']
    s3_pred_dict = s3_matches.groupby('id1')['id2'].apply(list).to_dict()
    for s1 in val_ids:
        if s1 not in s3_pred_dict: s3_pred_dict[s1] = []
        
    f05_s3, p_s3, r_s3 = calculate_f05_score(val_true_dict, s3_pred_dict)
    
    print("\n" + "=" * 85)
    print(" FINAL 3-STAGE MULTI-MODEL CASCADE RESULTS COMPARISON")
    print("=" * 85)
    print(f" Metric        | Stage 1 Baseline | Stage 1+2 Cascade | Stage 1+2+3 Complete | Net Gain")
    print(f" --------------+------------------+-------------------+----------------------+----------")
    print(f" Macro F0.5    | {f05_s1:16.4f} | {f05_casc:17.4f} | {f05_s3:20.4f} | {f05_s3 - f05_s1:+8.4f}")
    print(f" Precision     | {p_s1:16.4f} | {p_casc:17.4f} | {p_s3:20.4f} | {p_s3 - p_s1:+8.4f}")
    print(f" Recall        | {r_s1:16.4f} | {r_casc:17.4f} | {r_s3:20.4f} | {r_s3 - r_s1:+8.4f}")
    print("=" * 85)
    
    # 10. Concrete Case Studies
    print("\n10. Real Case Studies from the Validation Set:")
    def s_str(text):
        return str(text).encode('ascii', errors='replace').decode('ascii')
    
    # Case A: Clear match
    case_a = df_val[(df_val['stage1_score'] >= 0.90) & (df_val['target'] == 1)].head(1)
    if len(case_a) > 0:
        row = case_a.iloc[0]
        print("\n [CASE A: Clear Match Accepted by Stage 1]")
        print(f"   Entity 1: {s_str(row['name1'])} | {s_str(row['addr1'])}")
        print(f"   Entity 2: {s_str(row['name2'])} | {s_str(row['addr2'])}")
        print(f"   S1 Score: {row['stage1_score']:.4f} -> Decision: {row['decision']} (Ground Truth: {row['target']})")
        
    # Case B: Clear reject
    case_b = df_val[(df_val['stage1_score'] <= 0.15) & (df_val['target'] == 0)].head(1)
    if len(case_b) > 0:
        row = case_b.iloc[0]
        print("\n [CASE B: False Candidate Filtered by Stage 1]")
        print(f"   Entity 1: {s_str(row['name1'])} | {s_str(row['addr1'])}")
        print(f"   Entity 2: {s_str(row['name2'])} | {s_str(row['addr2'])}")
        print(f"   S1 Score: {row['stage1_score']:.4f} -> Decision: {row['decision']} (Ground Truth: {row['target']})")
        
    # Case C: Borderline resolved as Match by Stage 2
    case_c = df_val[stage2_mask & (df_val['decision'] == 'MATCH') & (df_val['target'] == 1)].head(1)
    if len(case_c) > 0:
        row = case_c.iloc[0]
        print("\n [CASE C: Borderline Match Rescued by Stage 2 Cross-Encoder]")
        print(f"   Entity 1: {s_str(row['name1'])} | {s_str(row['addr1'])}")
        print(f"   Entity 2: {s_str(row['name2'])} | {s_str(row['addr2'])}")
        print(f"   S1 Score: {row['stage1_score']:.4f} | S2 Score: {row['stage2_score']:.4f} -> Decision: {row['decision']} (Ground Truth: {row['target']})")
        
    # Case D: Borderline resolved as Reject by Stage 2
    case_d = df_val[stage2_mask & (df_val['decision'] == 'REJECT') & (df_val['target'] == 0)].head(1)
    if len(case_d) > 0:
        row = case_d.iloc[0]
        print("\n [CASE D: Ambiguous False Candidate Rejected by Stage 2]")
        print(f"   Entity 1: {s_str(row['name1'])} | {s_str(row['addr1'])}")
        print(f"   Entity 2: {s_str(row['name2'])} | {s_str(row['addr2'])}")
        print(f"   S1 Score: {row['stage1_score']:.4f} | S2 Score: {row['stage2_score']:.4f} -> Decision: {row['decision']} (Ground Truth: {row['target']})")

    # Case E: High-ambiguity pair arbitrated by Stage 3 Supreme Judge
    if s3_mask.sum() > 0:
        arbitrated = df_val[s3_mask].head(2)
        print("\n [CASE E: High-Ambiguity Pairs Arbitrated by Stage 3 Supreme Judge]")
        for idx, row in arbitrated.iterrows():
            print(f"   - Entity 1: {s_str(row['name1'])} | {s_str(row['addr1'])}")
            print(f"     Entity 2: {s_str(row['name2'])} | {s_str(row['addr2'])}")
            print(f"     S1 Score: {row['stage1_score']:.4f} | S2 Score: {row['stage2_score']:.4f} -> S3 Decision: {row['decision_s3']} (Ground Truth: {row['target']})")
            
    print("\nFull 3-Stage Multi-Model Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
