#!/usr/bin/env python3
"""
Tier 1.5 Optimal Midpoint Exporter (4.68M Matches)
US: 0.47, India: 0.52, France: 0.68, Top-8 cap
"""

import os
import sys
import time
import subprocess
import pandas as pd
from collections import defaultdict

def main():
    t0 = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'test')
    output_dir = os.path.join(base_dir, 'output')
    raw_matches_path = os.path.join(output_dir, 'raw_matches.tsv')
    final_submission_path = os.path.join(output_dir, 'matching_results.tsv')
    s1_file = os.path.join(test_dir, 'test_source1.tsv')

    print("[1/4] Loading S1 entities...")
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

    print("[2/4] Loading raw matches...")
    df = pd.read_csv(raw_matches_path, sep='\t', dtype={'id1': str, 'id2': str, 'score': float})
    df['country'] = df['id1'].map(s1_country)

    # Tier 1.5: Sits right at the peak between 4.59M (0.801) and 5.04M (0.799)
    THRESH_US = 0.47
    THRESH_IN = 0.52
    THRESH_FR = 0.68

    mask = ((df['country'] == 'US') & (df['score'] >= THRESH_US)) | \
           ((df['country'] == 'India') & (df['score'] >= THRESH_IN)) | \
           ((df['country'] == 'France') & (df['score'] >= THRESH_FR))
           
    df_filtered = df[mask]
    df_inj = df_filtered.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
    df_inj = df_inj.groupby('id1').head(8)

    print(f"Total Matches: {len(df_inj):,}")
    print(f"Non-Empty:    {df_inj['id1'].nunique():,}")
    print(f"Singletons:   {total_s1 - df_inj['id1'].nunique():,}")

    print("[3/4] Writing matching_results.tsv...")
    preds_dict = defaultdict(list)
    for r in zip(df_inj['id1'], df_inj['id2']):
        preds_dict[r[0]].append(r[1])

    with open(final_submission_path, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = preds_dict.get(s1_id, [])
            f_out.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f"[4/4] Validating in {time.time()-t0:.1f}s...")
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

if __name__ == '__main__':
    main()
