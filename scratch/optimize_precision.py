import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, time
from collections import defaultdict
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))
from evaluate_local import compute_macro_f05, print_evaluation_report

def main():
    print("=" * 80)
    print(" FAST PRECISION & SINGLETON OPTIMIZER (0.5s Grid Search via Cache)")
    print("=" * 80)

    val_gt_file = os.path.join(base_dir, 'data', 'processed', 'val_ground_truth.tsv')
    cache_path = os.path.join(base_dir, 'data', 'processed', 'val_cache_df_scores.parquet')

    if not os.path.exists(cache_path):
        print(f"Error: Cache {cache_path} does not exist yet. Please wait for eval_local_exact.py to finish.")
        return

    # 1. Load Ground Truth
    gt_dict = {}
    with open(val_gt_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            ms = p[1].split(',') if len(p) > 1 and p[1].strip() else []
            gt_dict[s1] = set(m.strip() for m in ms if m.strip())
    all_s1_ids = list(gt_dict.keys())

    # 2. Load Cached Predictions
    t0 = time.time()
    df_raw = pd.read_parquet(cache_path)
    print(f"Loaded {len(df_raw):,} precomputed predictions in {time.time()-t0:.2f}s.")

    # Grid search parameters
    base_thresholds = [0.35, 0.38, 0.40, 0.42, 0.45, 0.50]
    bldg_cuts = [75, 80, 85, 90]
    single_margins = [0.0, 0.05, 0.10, 0.15] # extra score required for isolated single matches

    best_score = -1.0
    best_config = None
    best_rep = None

    print("\nStarting Grid Search...")
    for b_cut in bldg_cuts:
        # Apply Building mismatch veto
        df_eval = df_raw.copy()
        bldg_mismatch = (df_eval['clean_st_mismatch'] == 1) & (df_eval['name_tsr'] < b_cut)
        df_eval.loc[bldg_mismatch, 'score'] = 0.0

        # Brand collision veto
        brand_col = (df_eval['brand_first'] < 0.35) & (df_eval['name_tsr'] < 50)
        df_eval.loc[brand_col, 'score'] = 0.0

        for t in base_thresholds:
            for sm in single_margins:
                # 1. Filter candidates at base threshold t
                matched = df_eval[df_eval['score'] >= t]
                
                # Injective deduplication
                inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
                
                # 2. Singleton Defense: if an S1 entity has only 1 match candidate, require score >= t + sm
                if sm > 0:
                    counts = inj['id1'].value_counts()
                    single_s1 = set(counts[counts == 1].index)
                    
                    # Drop isolated matches that fail the margin
                    mask_drop = inj['id1'].isin(single_s1) & (inj['score'] < (t + sm))
                    inj = inj[~mask_drop]

                preds_dict = defaultdict(list)
                for _, row in inj.iterrows():
                    preds_dict[row['id1']].append(row['id2'])

                res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1_ids)
                
                if res['macro_f05'] > best_score:
                    best_score = res['macro_f05']
                    best_config = (t, b_cut, sm)
                    best_rep = res
                    print(f" NEW BEST: Macro F0.5 = {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | S-Acc: {res['singleton_accuracy']*100:.1f}% (T={t:.2f}, BldgCut={b_cut}, S-Margin=+{sm:.2f})")

    print("\n" + "=" * 80)
    print(f" OPTIMAL CONFIGURATION: Threshold={best_config[0]:.2f}, BldgCut={best_config[1]}, SingleMargin=+{best_config[2]:.2f}")
    print(f" BEST MACRO F0.5: {best_score:.4f}")
    print("=" * 80)
    print_evaluation_report(best_rep, title="OPTIMIZED VALIDATION BENCHMARK REPORT")

if __name__ == '__main__':
    main()
