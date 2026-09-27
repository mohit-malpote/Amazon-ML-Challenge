#!/usr/bin/env python3
"""
Incremental Push Engine (Targeting 0.806+):
1. Rescues empty S1 entities with ML score >= 0.40 from raw_matches.tsv (injective 1-to-1).
2. Rescues remaining empty S1 entities with exact rare business names (cnt <= 3, len >= 6).
3. Backs up 0.805 baseline and runs official validator.
"""

import os
import sys
import time
import shutil
import subprocess
import pandas as pd
from collections import Counter, defaultdict

def main():
    t0 = time.time()
    print("=" * 80)
    print(" EXECUTING INCREMENTAL PUSH (Targeting 0.806+)")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'test')
    output_dir = os.path.join(base_dir, 'output')
    matching_file = os.path.join(output_dir, 'matching_results.tsv')
    candidate_file = os.path.join(output_dir, 'candidate_pairs.tsv')
    raw_file = os.path.join(output_dir, 'raw_matches.tsv')
    s1_file = os.path.join(test_dir, 'test_source1.tsv')

    # 0. Backup 0.805 baseline
    bak_file = matching_file + '.bak_0805'
    print(f"[0/5] Creating safe backup of 0.805 baseline at {bak_file}...")
    shutil.copyfile(matching_file, bak_file)

    # 1. Load current 0.805 predictions
    print("\n[1/5] Loading current 0.805 predictions...")
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

    # 2. Rescue from raw_matches with score >= 0.40
    print("\n[2/5] Rescuing empty entities with ML candidate score >= 0.40...")
    df_raw = pd.read_csv(raw_file, sep='\t', dtype={'id1': str, 'id2': str, 'score': float})
    df_empty = df_raw[df_raw['id1'].isin(empty_s1)]
    df_empty_avail = df_empty[~df_empty['id2'].isin(cand_assigned)]
    
    df_rescued_ml = df_empty_avail[df_empty_avail['score'] >= 0.40].sort_values('score', ascending=False)
    df_rescued_ml = df_rescued_ml.drop_duplicates(subset=['id2'], keep='first')
    df_rescued_ml = df_rescued_ml.drop_duplicates(subset=['id1'], keep='first')
    
    ml_count = len(df_rescued_ml)
    print(f"   -> ML Rescued: {ml_count:,} entities (score >= 0.40)!")
    for r in zip(df_rescued_ml['id1'], df_rescued_ml['id2']):
        s1_id, cid = r[0], r[1]
        current_preds[s1_id] = [cid]
        cand_assigned.add(cid)
        empty_s1.remove(s1_id)

    # 3. Rescue remaining empty S1 with rare exact names (cnt <= 3, len >= 6)
    print("\n[3/5] Rescuing remaining empty entities with rare exact names (cnt <= 3)...")
    name_counts = Counter()
    cand_by_name = defaultdict(list)
    for src in ['test_source2.tsv', 'test_source3.tsv']:
        src_path = os.path.join(test_dir, src)
        with open(src_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                cid = p[0]
                nm = p[1].strip().lower() if len(p) > 1 and p[1] else ''
                ct = p[3].strip() if len(p) > 3 and p[3] else ''
                if len(nm) >= 6:
                    key = (nm, ct)
                    name_counts[key] += 1
                    if len(cand_by_name[key]) < 4:
                        cand_by_name[key].append(cid)

    new_candidate_additions = defaultdict(list)
    name_rescued_count = 0
    with open(s1_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1_id = p[0]
            if s1_id in empty_s1:
                nm = p[1].strip().lower() if len(p) > 1 and p[1] else ''
                ct = p[3].strip() if len(p) > 3 and p[3] else ''
                if len(nm) >= 6:
                    key = (nm, ct)
                    cnt = name_counts.get(key, 0)
                    if 1 <= cnt <= 3:
                        cands = [c for c in cand_by_name[key] if c not in cand_assigned]
                        s2_c = [c for c in cands if c.startswith('S2-')]
                        s3_c = [c for c in cands if c.startswith('S3-')]
                        chosen = []
                        if s2_c and s3_c:
                            chosen = [s2_c[0], s3_c[0]]
                        elif s2_c:
                            chosen = [s2_c[0]]
                        elif s3_c:
                            chosen = [s3_c[0]]
                        if chosen:
                            current_preds[s1_id] = chosen
                            new_candidate_additions[s1_id] = chosen
                            cand_assigned.update(chosen)
                            empty_s1.remove(s1_id)
                            name_rescued_count += 1

    print(f"   -> Exact Name Rescued: {name_rescued_count:,} entities!")
    print(f"   -> Total Rescued in this pass: {ml_count + name_rescued_count:,}")
    print(f"   -> New Non-Empty Count:        {total_s1 - len(empty_s1):,} ({(total_s1 - len(empty_s1))/total_s1*100:.2f}%)")
    print(f"   -> Remaining Singletons:       {len(empty_s1):,} ({len(empty_s1)/total_s1*100:.2f}%)")

    # 4. Synchronize candidate_pairs.tsv if name matches added new IDs
    if new_candidate_additions:
        print("\n[4/5] Synchronizing candidate_pairs.tsv...")
        temp_cands = candidate_file + ".tmp"
        with open(candidate_file, 'r', encoding='utf-8') as f_in, open(temp_cands, 'w', encoding='utf-8') as f_out:
            f_out.write(f_in.readline())
            for line in f_in:
                p = line.rstrip('\n').split('\t')
                s1_id = p[0]
                new_c = new_candidate_additions.get(s1_id, [])
                if new_c:
                    existing = p[1].split(',') if len(p) > 1 and p[1].strip() else []
                    combined = list(dict.fromkeys(new_c + existing))
                    f_out.write(f"{s1_id}\t{','.join(combined)}\n")
                else:
                    f_out.write(line)
        shutil.move(temp_cands, candidate_file)
        print("   -> candidate_pairs.tsv synchronized.")

    # 5. Export Updated matching_results.tsv
    print("\n[5/5] Exporting updated matching_results.tsv...")
    with open(matching_file, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        with open(s1_file, 'r', encoding='utf-8') as f_s1:
            next(f_s1)
            for line in f_s1:
                s1_id = line.partition('\t')[0]
                ms = current_preds.get(s1_id, [])
                f_out.write(f"{s1_id}\t{','.join(ms)}\n")

    print(f"   -> Export completed in {time.time()-t0:.1f}s.")

    # 6. Run Official Validator
    print("\nRunning official submission validator...")
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
