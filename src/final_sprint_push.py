#!/usr/bin/env python3
"""
Final Sprint Push Engine:
Rescues empty S1 entities with ML candidate score >= 0.38 (injective 1-to-1).
Ultra fast (<30s runtime) for the final pre-midnight submission.
"""

import os
import sys
import time
import shutil
import subprocess
import pandas as pd

def main():
    t0 = time.time()
    print("=" * 80)
    print(" EXECUTING FINAL SPRINT PUSH (Last Pre-Midnight Submission)")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'test')
    output_dir = os.path.join(base_dir, 'output')
    matching_file = os.path.join(output_dir, 'matching_results.tsv')
    candidate_file = os.path.join(output_dir, 'candidate_pairs.tsv')
    raw_file = os.path.join(output_dir, 'raw_matches.tsv')
    s1_file = os.path.join(test_dir, 'test_source1.tsv')

    # 0. Backup current baseline
    bak_file = matching_file + '.bak_prefinal'
    print(f"[0/4] Creating safe backup at {bak_file}...")
    shutil.copyfile(matching_file, bak_file)

    # 1. Load current predictions
    print("\n[1/4] Loading current predictions...")
    current_preds = {}
    cand_assigned = set()
    with open(matching_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            ms = [m.strip() for m in p[1].split(',') if m.strip()] if len(p) > 1 and p[1].strip() else []
            current_preds[s1] = ms
            for m in ms:
                cand_assigned.add(m)

    total_s1 = len(current_preds)
    empty_s1 = set(k for k, v in current_preds.items() if len(v) == 0)
    print(f"   -> Total S1 entities:         {total_s1:,}")
    print(f"   -> Non-empty entities:       {total_s1 - len(empty_s1):,}")
    print(f"   -> Currently empty entities:   {len(empty_s1):,}")

    # 2. Rescue from raw_matches with score >= 0.38
    print("\n[2/4] Rescuing empty entities with ML candidate score >= 0.38...")
    df_raw = pd.read_csv(raw_file, sep='\t', dtype={'id1': str, 'id2': str, 'score': float})
    df_empty = df_raw[df_raw['id1'].isin(empty_s1)]
    df_empty_avail = df_empty[~df_empty['id2'].isin(cand_assigned)]
    
    SCORE_THRESH = 0.38
    df_rescued_ml = df_empty_avail[df_empty_avail['score'] >= SCORE_THRESH].sort_values('score', ascending=False)
    df_rescued_ml = df_rescued_ml.drop_duplicates(subset=['id2'], keep='first')
    df_rescued_ml = df_rescued_ml.drop_duplicates(subset=['id1'], keep='first')
    
    ml_count = len(df_rescued_ml)
    print(f"   -> ML Rescued: {ml_count:,} entities (score >= {SCORE_THRESH:.2f})!")
    for r in zip(df_rescued_ml['id1'], df_rescued_ml['id2']):
        s1_id, cid = r[0], r[1]
        current_preds[s1_id] = [cid]
        cand_assigned.add(cid)
        empty_s1.remove(s1_id)

    print(f"   -> New Non-Empty Count:        {total_s1 - len(empty_s1):,} ({(total_s1 - len(empty_s1))/total_s1*100:.2f}%)")
    print(f"   -> Remaining Singletons:       {len(empty_s1):,} ({len(empty_s1)/total_s1*100:.2f}%)")

    # 3. Export Updated matching_results.tsv
    print("\n[3/4] Exporting updated matching_results.tsv...")
    with open(matching_file, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        with open(s1_file, 'r', encoding='utf-8') as f_s1:
            next(f_s1)
            for line in f_s1:
                s1_id = line.partition('\t')[0]
                ms = current_preds.get(s1_id, [])
                f_out.write(f"{s1_id}\t{','.join(ms)}\n")

    print(f"   -> Export completed in {time.time()-t0:.1f}s.")

    # 4. Run Official Validator
    print("\n[4/4] Running official submission validator...")
    validator_script = os.path.join(base_dir, 'student_resource', 'utils', 'validate_submission.py')
    cmd = [
        sys.executable, validator_script,
        '--matching', matching_file,
        '--candidate', candidate_file,
        '--test-dir', test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator STDERR:", res.stderr)

    print("=" * 80)
    print(f" ALL OPERATIONS COMPLETED IN {(time.time()-t0):.1f} SECONDS!")
    print(f" New submission file ready: {matching_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()
