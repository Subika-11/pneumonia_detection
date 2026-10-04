import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

METADATA_CSV = r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\data\processed\rsna_metadata.csv"
OUTPUT_SPLITS_CSV = r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\data\processed\splits.csv"
RANDOM_SEED = 42

def create_leakage_free_splits(df, seed=42):
    """
    Creates patient-level, stratified Train/Val/Test splits (80/10/10).
    Ensures that all images from the same patient strictly remain in the same partition.
    """
    print("=" * 60)
    print("PHASE 4: LEAKAGE-FREE PATIENT-LEVEL DATA SPLITTING")
    print("=" * 60)
    print(f"Total cohort size: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    
    unique_patients = df['patient_id'].nunique()
    total_records = len(df)
    
    if unique_patients == total_records:
        print("Note: In RSNA 2018 adjudicated cohort, each study corresponds to a distinct patient ID (1:1 mapping).")
    else:
        print(f"Warning: Multiple studies per patient detected ({total_records - unique_patients} repeated patients). Grouping by patient_id to prevent leakage.")
        
    # We use StratifiedGroupKFold (10 folds) to partition into 80% train (8 folds), 10% val (1 fold), 10% test (1 fold)
    # This guarantees zero patient overlap even if patient IDs repeat!
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    
    splits_dict = {}
    folds = np.zeros(len(df), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(df, df['target'], groups=df['patient_id'])):
        folds[test_idx] = fold
        
    # Fold 0 = Test (10%), Fold 1 = Validation (10%), Folds 2..9 = Train (80%)
    split_col = []
    for f in folds:
        if f == 0:
            split_col.append('test')
        elif f == 1:
            split_col.append('val')
        else:
            split_col.append('train')
            
    df['split'] = split_col
    
    # Verification of zero leakage
    train_patients = set(df[df['split'] == 'train']['patient_id'])
    val_patients = set(df[df['split'] == 'val']['patient_id'])
    test_patients = set(df[df['split'] == 'test']['patient_id'])
    
    overlap_train_val = train_patients & val_patients
    overlap_train_test = train_patients & test_patients
    overlap_val_test = val_patients & test_patients
    
    print("\n--- PATIENT OVERLAP VERIFICATION ---")
    print(f"Train / Val patient overlap: {len(overlap_train_val)} (Must be 0)")
    print(f"Train / Test patient overlap: {len(overlap_train_test)} (Must be 0)")
    print(f"Val / Test patient overlap: {len(overlap_val_test)} (Must be 0)")
    assert len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0, "DATA LEAKAGE DETECTED!"
    print("Verification Passed: 100% Leakage-Free Partitions Guaranteed.")
    
    # Split statistics
    print("\n--- SPLIT DISTRIBUTION SUMMARY ---")
    summary = []
    for s in ['train', 'val', 'test']:
        sub = df[df['split'] == s]
        n_total = len(sub)
        n_pos = (sub['target'] == 1).sum()
        n_neg = (sub['target'] == 0).sum()
        pct_pos = (n_pos / n_total) * 100
        summary.append({
            'Split': s.upper(),
            'Total Images': n_total,
            'Pneumonia (Pos)': n_pos,
            'Non-Pneumonia (Neg)': n_neg,
            'Pneumonia %': f"{pct_pos:.2f}%",
            'Fraction of Dataset': f"{n_total / len(df) * 100:.1f}%"
        })
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    
    return df

def main():
    df = pd.read_csv(METADATA_CSV)
    df_splits = create_leakage_free_splits(df, seed=RANDOM_SEED)
    
    # Save the splits table
    output_cols = ['study_uid', 'patient_id', 'zip_entry', 'target', 'class_name', 'num_boxes', 
                   'patient_age', 'patient_sex', 'view_position', 'modality', 'split']
    df_splits[output_cols].to_csv(OUTPUT_SPLITS_CSV, index=False)
    print(f"\nSaved split assignments to: {OUTPUT_SPLITS_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()
