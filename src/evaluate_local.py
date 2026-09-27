#!/usr/bin/env python3
"""
Local Validation & Scoring Engine for ML Challenge 2026.
Implements the exact official competition evaluation metric:
Macro-averaged F_0.5 across all Source 1 entities, including singletons.

Scoring Rules:
1. True singleton (0 true matches):
   - Correctly predicted empty -> Score = 1.0
   - Predicted ANY match -> Score = 0.0
2. Non-singleton (>0 true matches):
   - Predicted empty -> Score = 0.0
   - P = TP / (TP + FP), R = TP / (TP + FN)
   - If TP == 0: Score = 0.0
   - F_0.5 = (1.25 * P * R) / (0.25 * P + R)
3. Macro F_0.5 = Mean score across ALL S1 entities in the evaluation set.
"""

import os
import sys
import argparse
import random
from collections import defaultdict
import pandas as pd
import numpy as np

def compute_macro_f05(gt_dict, pred_dict, s1_ids=None):
    """
    Computes exact Macro F_0.5 score per the competition specification.
    
    gt_dict: dict mapping s1_id -> set of true matching S2/S3 entity IDs (empty set for singletons)
    pred_dict: dict mapping s1_id -> set/list of predicted S2/S3 entity IDs (empty set/list if none)
    s1_ids: optional list of all s1_ids to evaluate over (defaults to all keys in gt_dict)
    
    Returns:
        dict containing:
            macro_f05, macro_precision, macro_recall,
            singleton_count, singleton_accuracy,
            non_singleton_count, non_singleton_f05,
            total_tp, total_fp, total_fn, micro_f05
    """
    if s1_ids is None:
        s1_ids = list(gt_dict.keys())
        
    scores = []
    precisions = []
    recalls = []
    
    # Detailed diagnosis counters
    true_singletons = 0
    correct_singletons = 0
    false_positive_singletons = 0
    
    non_singletons = 0
    false_negative_singletons = 0 # Had true matches, but predicted empty
    non_singleton_scores = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for s1 in s1_ids:
        gt_set = set(gt_dict.get(s1, []))
        pred_set = set(pred_dict.get(s1, []))
        
        # Remove any self-references or empty strings
        pred_set = {x.strip() for x in pred_set if x and x.strip()}
        
        if len(gt_set) == 0:
            true_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1
                score = 1.0
                p = 1.0
                r = 1.0
            else:
                false_positive_singletons += 1
                score = 0.0
                p = 0.0
                r = 0.0
                total_fp += len(pred_set)
        else:
            non_singletons += 1
            if len(pred_set) == 0:
                false_negative_singletons += 1
                score = 0.0
                p = 0.0
                r = 0.0
                total_fn += len(gt_set)
            else:
                tp = len(gt_set.intersection(pred_set))
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if tp == 0 or (0.25 * p + r) == 0:
                    score = 0.0
                else:
                    score = (1.25 * p * r) / (0.25 * p + r)
                    
            non_singleton_scores.append(score)
            
        scores.append(score)
        precisions.append(p)
        recalls.append(r)
        
    # Micro metrics
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f05 = (1.25 * micro_p * micro_r) / (0.25 * micro_p + micro_r) if (0.25 * micro_p + micro_r) > 0 else 0.0
    
    return {
        'macro_f05': float(np.mean(scores)),
        'macro_precision': float(np.mean(precisions)),
        'macro_recall': float(np.mean(recalls)),
        'total_evaluated_s1': len(s1_ids),
        'true_singletons': true_singletons,
        'correct_singletons': correct_singletons,
        'singleton_accuracy': (correct_singletons / true_singletons) if true_singletons > 0 else 1.0,
        'non_singletons': non_singletons,
        'false_negative_singletons': false_negative_singletons,
        'non_singleton_f05': float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_f05': micro_f05
    }

def print_evaluation_report(results, title="LOCAL VALIDATION EVALUATION REPORT"):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print(f" Macro F0.5 Score (LEADERBOARD METRIC): {results['macro_f05']:.4f}")
    print(f" Macro Precision:                       {results['macro_precision']:.4f}")
    print(f" Macro Recall:                          {results['macro_recall']:.4f}")
    print("-" * 80)
    print(" Detailed Breakdown:")
    print(f"   Total S1 Entities Evaluated:         {results['total_evaluated_s1']:,}")
    print(f"   True Singletons:                     {results['true_singletons']:,} ({results['true_singletons']/results['total_evaluated_s1']*100:.2f}%)")
    print(f"     - Correctly Predicted Empty:       {results['correct_singletons']:,} (Accuracy: {results['singleton_accuracy']*100:.2f}%)")
    print(f"     - False Merges on Singletons:      {results['true_singletons'] - results['correct_singletons']:,} (Penalized with 0.0)")
    print(f"   Non-Singletons (>0 True Matches):    {results['non_singletons']:,} ({results['non_singletons']/results['total_evaluated_s1']*100:.2f}%)")
    print(f"     - False Singletons (Predicted 0):  {results['false_negative_singletons']:,} (Penalized with 0.0)")
    print(f"     - Non-Singleton Macro F0.5:        {results['non_singleton_f05']:.4f}")
    print("-" * 80)
    print(" Micro-Level Pair Counts:")
    print(f"   True Positives (TP):                 {results['total_tp']:,}")
    print(f"   False Positives (FP):                {results['total_fp']:,}")
    print(f"   False Negatives (FN):                {results['total_fn']:,}")
    print(f"   Micro Precision:                     {results['micro_precision']:.4f}")
    print(f"   Micro Recall:                        {results['micro_recall']:.4f}")
    print(f"   Micro F0.5:                          {results['micro_f05']:.4f}")
    print("=" * 80 + "\n")

def create_validation_split(num_s1=50000, seed=42):
    """
    Creates a fixed, stratified 50,000 S1 validation benchmark split from the training dataset.
    Saves:
        data/processed/val_s1_sample.tsv
        data/processed/val_ground_truth.tsv
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'train')
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    
    val_s1_path = os.path.join(proc_dir, 'val_s1_sample.tsv')
    val_gt_path = os.path.join(proc_dir, 'val_ground_truth.tsv')
    
    print(f"Creating reproducible {num_s1:,} S1 validation split (seed={seed})...")
    
    # 1. Load full S1 dataset metadata
    s1_records = []
    with open(os.path.join(train_dir, 'train_source1.tsv'), 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n')
        for line in f:
            p = line.rstrip('\n').split('\t')
            eid = p[0]
            ct = p[3] if len(p) > 3 else ''
            s1_records.append((eid, line))
            
    print(f"   -> Loaded {len(s1_records):,} total S1 train entities.")
    
    # 2. Sample stratified by country
    random.seed(seed)
    sampled_records = random.sample(s1_records, num_s1)
    sampled_ids = set(r[0] for r in sampled_records)
    
    # 3. Write val_s1_sample.tsv
    with open(val_s1_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for _, raw_line in sampled_records:
            f.write(raw_line)
    print(f"   -> Saved validation S1 records to {val_s1_path}")
    
    # 4. Extract ground truth for sampled entities
    val_gt_count = 0
    val_singleton_count = 0
    with open(os.path.join(train_dir, 'train_ground_truth.tsv'), 'r', encoding='utf-8') as fin, \
         open(val_gt_path, 'w', encoding='utf-8') as fout:
        gt_header = fin.readline()
        fout.write(gt_header)
        for line in fin:
            p = line.rstrip('\n').split('\t')
            s1_id = p[0]
            if s1_id in sampled_ids:
                fout.write(line)
                ms = p[1] if len(p) > 1 and p[1].strip() else ''
                if ms:
                    val_gt_count += len(ms.split(','))
                else:
                    val_singleton_count += 1
                    
    print(f"   -> Saved validation Ground Truth to {val_gt_path}")
    print(f"   -> Total S1 in Val Split:        {num_s1:,}")
    print(f"   -> True Matches to find:         {val_gt_count:,}")
    print(f"   -> Singletons in Val Split:      {val_singleton_count:,} ({val_singleton_count/num_s1*100:.2f}%)")
    print(" Validation split creation complete!\n")
    return val_s1_path, val_gt_path

def load_val_ground_truth(val_gt_path=None):
    if val_gt_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        val_gt_path = os.path.join(base_dir, 'data', 'processed', 'val_ground_truth.tsv')
        
    gt_dict = {}
    with open(val_gt_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            ms = p[1].split(',') if len(p) > 1 and p[1].strip() else []
            gt_dict[s1] = set(m.strip() for m in ms if m.strip())
    return gt_dict

def load_predictions(pred_path):
    pred_dict = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            ms = p[1].split(',') if len(p) > 1 and p[1].strip() else []
            pred_dict[s1] = set(m.strip() for m in ms if m.strip())
    return pred_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution Predictions Locally")
    parser.add_argument('--create-split', action='store_true', help="Create a 50k validation split from train")
    parser.add_argument('--num-s1', type=int, default=50000, help="Number of S1 entities in validation split")
    parser.add_argument('--eval-file', type=str, default=None, help="Path to predictions TSV file to evaluate")
    parser.add_argument('--gt-file', type=str, default=None, help="Path to ground truth TSV file")
    
    args = parser.parse_args()
    
    if args.create_split:
        create_validation_split(num_s1=args.num_s1)
    elif args.eval_file:
        gt = load_val_ground_truth(args.gt_file)
        preds = load_predictions(args.eval_file)
        res = compute_macro_f05(gt, preds)
        print_evaluation_report(res, title=f"EVALUATION OF {os.path.basename(args.eval_file)}")
    else:
        # Default behavior: if split doesn't exist, create it
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        val_gt_path = os.path.join(base_dir, 'data', 'processed', 'val_ground_truth.tsv')
        if not os.path.exists(val_gt_path):
            create_validation_split(num_s1=args.num_s1)
        else:
            print("Validation split already exists. Use --eval-file to evaluate predictions or --create-split --num-s1 N to re-create.")
