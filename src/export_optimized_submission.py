#!/usr/bin/env python3
"""
Ultra-Fast High-Precision Submission Exporter (Runs in ~30 seconds)
Applies:
1. Country-Calibrated Thresholds (US: 0.52, India: 0.56, France: 0.76)
2. Injective 1-to-1 Mapping Deduplication
3. Dynamic Margin Pruning (Discards weak trailing candidates < 0.60 of top match)
4. Top-6 Maximum Match Cap per S1 entity
"""

import os
import sys
import time
import subprocess
import pandas as pd
from collections import defaultdict

def main():
    t0 = time.time()
    print("=" * 80)
    print(" EXECUTING BLENDED HIGH-PRECISION CEILING EXPORTER")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'test')
    output_dir = os.path.join(base_dir, 'output')
    raw_matches_path = os.path.join(output_dir, 'raw_matches.tsv')
    final_submission_path = os.path.join(output_dir, 'matching_results.tsv')
    s1_file = os.path.join(test_dir, 'test_source1.tsv')

    # 1. Load S1 IDs and Country Metadata
    print("\n[1/5] Loading Test S1 Entities and Country Metadata...")
    all_s1_ids = []
    s1_country = {}
    with open(s1_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            eid = p[0]
            all_s1_ids.append(eid)
            s1_country[eid] = p[3] if len(p) > 3 else 'Unknown'
            
    total_s1 = len(all_s1_ids)
    print(f"   -> Loaded {total_s1:,} Test S1 entities.")

    # 2. Load Scored Candidate Pairs
    print(f"\n[2/5] Loading pre-scored predictions from {raw_matches_path}...")
    df = pd.read_csv(raw_matches_path, sep='\t', dtype={'id1': str, 'id2': str, 'score': float})
    print(f"   -> Loaded {len(df):,} candidate pairs on disk.")

    # 3. Country-Calibrated Thresholding
    print("\n[3/5] Applying Country-Calibrated Precision Filtering...")
    THRESH_US = 0.52
    THRESH_IN = 0.56
    THRESH_FR = 0.76

    print(f"   -> US Threshold:     {THRESH_US:.2f}")
    print(f"   -> India Threshold:  {THRESH_IN:.2f}")
    print(f"   -> France Threshold: {THRESH_FR:.2f}")

    df['country'] = df['id1'].map(s1_country)
    mask = ((df['country'] == 'US') & (df['score'] >= THRESH_US)) | \
           ((df['country'] == 'India') & (df['score'] >= THRESH_IN)) | \
           ((df['country'] == 'France') & (df['score'] >= THRESH_FR))
           
    filtered_df = df[mask]
    print(f"   -> Retained {len(filtered_df):,} candidates meeting country thresholds.")

    # 4. Injective 1-to-1 Deduplication + Dynamic Margin Pruning + Top-6 Cap
    print("\n[4/5] Enforcing Injective 1-to-1 Mapping + Dynamic Margin Pruning...")
    # Injective: keep highest score per S2/S3 candidate id2
    df_inj = filtered_df.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first').copy()
    
    # Dynamic Margin Pruning: Discard weak tail candidates when a high-confidence match exists
    df_inj['max_s'] = df_inj.groupby('id1')['score'].transform('max')
    margin_mask = (df_inj['max_s'] < 0.75) | (df_inj['score'] >= df_inj['max_s'] * 0.60)
    df_inj = df_inj[margin_mask]
    
    # Cap to Top-6 matches per entity
    df_inj = df_inj.groupby('id1').head(6)
    
    total_matches = len(df_inj)
    non_empty_count = df_inj['id1'].nunique()
    singleton_count = total_s1 - non_empty_count
    
    print(f"   -> Final high-precision match pairs: {total_matches:,}")
    print(f"   -> Non-empty S1 entities:            {non_empty_count:,} ({non_empty_count/total_s1*100:.2f}%)")
    print(f"   -> Singleton S1 entities:            {singleton_count:,} ({singleton_count/total_s1*100:.2f}%)")
    print(f"   -> Avg matches per non-empty S1:     {total_matches/non_empty_count:.2f}")

    # 5. Format and Export to matching_results.tsv
    print(f"\n[5/5] Exporting formatted submission to {final_submission_path}...")
    preds_dict = defaultdict(list)
    for r in zip(df_inj['id1'], df_inj['id2']):
        preds_dict[r[0]].append(r[1])

    with open(final_submission_path, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = preds_dict.get(s1_id, [])
            f_out.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f"   -> Successfully wrote {total_s1:,} rows to {final_submission_path} in {time.time()-t0:.2f}s!")

    # 6. Run Official Submission Validator
    print("\nRunning official submission validator...")
    validator_script = os.path.join(base_dir, 'student_resource', 'utils', 'validate_submission.py')
    cmd = [
        sys.executable, validator_script,
        '--matching', final_submission_path,
        '--candidate', os.path.join(output_dir, 'candidate_pairs.tsv'),
        '--test-dir', test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator STDERR:", res.stderr)

    print("=" * 80)
    print(f" ALL TASKS COMPLETED SUCCESSFULLY IN {(time.time()-t0):.1f} SECONDS!")
    print(f" Optimized file ready for submission: {final_submission_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()
