#!/usr/bin/env python3
"""
Merges country candidate files into the final candidate_pairs.tsv and validates with official validator.
Usage:
  python src/merge_candidates.py
"""

import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    test_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'test')
    
    countries = ['India', 'US', 'France']
    files_to_merge = [os.path.join(output_dir, f"candidate_pairs_{c}.tsv") for c in countries]
    
    missing = [f for f in files_to_merge if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing candidate files: {missing}")
        print("Please ensure all 3 people have completed their country blocking script.")
        sys.exit(1)
        
    final_cands_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    final_test_cands_path = os.path.join(output_dir, 'candidate_pairs_test.tsv')
    
    print(f"Merging {len(files_to_merge)} country files into {final_cands_path}...")
    total_lines = 0
    with open(final_cands_path, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for c_file in files_to_merge:
            with open(c_file, 'r', encoding='utf-8') as f_in:
                header = f_in.readline() # Skip country header
                for line in f_in:
                    f_out.write(line)
                    total_lines += 1
                    
    # Also copy/symlink to candidate_pairs_test.tsv
    import shutil
    shutil.copyfile(final_cands_path, final_test_cands_path)
    
    print(f"Merge complete! Wrote {total_lines:,} entity rows to:")
    print(f"  - {final_cands_path}")
    print(f"  - {final_test_cands_path}\n")
    
    # Run official validator
    validator_path = os.path.join(base_dir, 'student_resource', 'utils', 'validate_submission.py')
    print("Running official submission validator on merged candidate file...")
    cmd = [
        sys.executable, validator_path,
        '--matching', os.path.join(output_dir, 'matching_results.tsv'),
        '--candidate', final_cands_path,
        '--test-dir', test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

if __name__ == '__main__':
    main()
