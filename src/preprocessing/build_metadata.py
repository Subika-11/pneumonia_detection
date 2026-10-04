import os
import sys
import json
import zipfile
import struct
import pandas as pd
from tqdm import tqdm

ZIP_PATH = r"D:\downloads\pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"
JSON_PATH = r"D:\downloads\pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
OUTPUT_METADATA_CSV = r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\data\processed\rsna_metadata.csv"
EXCLUSIONS_LOG_CSV = r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\data\processed\exclusions_log.csv"

def extract_dicom_tags(dcm_bytes):
    """Fast extraction of key DICOM header tags without loading pixel array."""
    if len(dcm_bytes) < 132 or dcm_bytes[128:132] != b'DICM':
        return None
    idx = 132
    info = {}
    while idx < min(len(dcm_bytes), 4096):
        try:
            group, element = struct.unpack('<HH', dcm_bytes[idx:idx+4])
            tag = (group, element)
            vr = dcm_bytes[idx+4:idx+6].decode('latin1', errors='ignore')
            if vr in ['OB', 'OD', 'OF', 'OL', 'OW', 'SQ', 'UC', 'UR', 'UT', 'UN']:
                length = struct.unpack('<I', dcm_bytes[idx+8:idx+12])[0]
                val_idx = idx + 12
            elif vr.isupper() and len(vr) == 2:
                length = struct.unpack('<H', dcm_bytes[idx+6:idx+8])[0]
                val_idx = idx + 8
            else:
                length = struct.unpack('<I', dcm_bytes[idx+4:idx+8])[0]
                val_idx = idx + 8
            val = dcm_bytes[val_idx:val_idx+length].decode('latin1', errors='ignore').strip('\x00').strip()
            
            if tag == (0x0010, 0x0020):
                info['patient_id'] = val
            elif tag == (0x0010, 0x0040):
                info['patient_sex'] = val
            elif tag == (0x0010, 0x1010):
                try:
                    info['patient_age'] = int(val)
                except:
                    info['patient_age'] = None
            elif tag == (0x0018, 0x5101):
                info['view_position'] = val
            elif tag == (0x0020, 0x000D):
                info['study_uid'] = val
            elif tag == (0x0020, 0x000E):
                info['series_uid'] = val
            elif tag == (0x0008, 0x0018):
                info['sop_uid'] = val
            elif tag == (0x0008, 0x0060):
                info['modality'] = val
            idx = val_idx + length
        except:
            break
    return info

def main():
    print("=" * 60)
    print("PHASE 3: BUILDING STANDARDIZED RSNA METADATA")
    print("=" * 60)
    
    # 1. Load and parse JSON annotations
    print(f"Loading annotations from: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    label_info = {}
    for lg in data.get('labelGroups', []):
        g_name = lg.get('name')
        for lbl in lg.get('labels', []):
            label_info[lbl.get('id')] = {'group': g_name, 'name': lbl.get('name')}
            
    annotations = data.get('datasets', [])[0].get('annotations', [])
    print(f"Total raw annotation records: {len(annotations)}")
    
    # Parse annotations by study
    study_consensus = {}
    study_boxes = {}
    non_calculated_studies = {}
    
    for a in annotations:
        s_uid = a.get('StudyInstanceUID')
        lid = a.get('labelId')
        linf = label_info.get(lid, {})
        grp = linf.get('group')
        lbl_name = linf.get('name')
        
        if grp == 'Calculated':
            study_consensus[s_uid] = lbl_name
            if lbl_name == 'Lung Opacity' and a.get('data') is not None:
                box = a.get('data')
                if s_uid not in study_boxes:
                    study_boxes[s_uid] = []
                study_boxes[s_uid].append([box.get('x'), box.get('y'), box.get('width'), box.get('height')])
        else:
            if s_uid not in non_calculated_studies:
                non_calculated_studies[s_uid] = []
            non_calculated_studies[s_uid].append(f"{grp}:{lbl_name}")
            
    print(f"Studies with verified consensus label: {len(study_consensus)}")
    print(f"Studies with bounding boxes: {len(study_boxes)}")
    
    # 2. Extract DICOM header metadata from ZIP
    print(f"\nReading DICOM headers from archive: {ZIP_PATH}")
    records = []
    exclusions = []
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        dcm_entries = [f for f in z.namelist() if f.endswith('.dcm')]
        print(f"Total DICOM entries in archive: {len(dcm_entries)}")
        
        for entry in tqdm(dcm_entries, desc="Extracting DICOM headers"):
            # Path components: .../<study_uid>/<series_uid>/<sop_uid>.dcm
            parts = entry.split('/')
            study_uid_from_path = parts[-3] if len(parts) >= 3 else ""
            
            # Read first 4KB of DICOM for header tags
            dcm_bytes = z.read(entry)
            info = extract_dicom_tags(dcm_bytes)
            
            if not info or not info.get('study_uid'):
                study_uid = study_uid_from_path
                patient_id = study_uid
                patient_age = None
                patient_sex = "Unknown"
                view_pos = "Unknown"
                modality = "CR"
            else:
                study_uid = info.get('study_uid', study_uid_from_path)
                patient_id = info.get('patient_id', study_uid)
                patient_age = info.get('patient_age')
                patient_sex = info.get('patient_sex', 'Unknown')
                view_pos = info.get('view_position', 'Unknown')
                modality = info.get('modality', 'CR')
                
            # Match with consensus label
            if study_uid in study_consensus:
                class_name = study_consensus[study_uid]
                # Binary target: 1 = Lung Opacity / Pneumonia, 0 = Non-pneumonia
                target = 1 if class_name == 'Lung Opacity' else 0
                boxes = json.dumps(study_boxes.get(study_uid, []))
                
                records.append({
                    'study_uid': study_uid,
                    'patient_id': patient_id,
                    'zip_entry': entry,
                    'target': target,
                    'class_name': class_name,
                    'num_boxes': len(study_boxes.get(study_uid, [])),
                    'boxes': boxes,
                    'patient_age': patient_age,
                    'patient_sex': patient_sex,
                    'view_position': view_pos,
                    'modality': modality,
                    'is_valid': True
                })
            else:
                reason = "No consensus label in Calculated group"
                other_labels = non_calculated_studies.get(study_uid, [])
                if other_labels:
                    reason += f" (Review tags: {'; '.join(other_labels[:3])})"
                exclusions.append({
                    'study_uid': study_uid,
                    'zip_entry': entry,
                    'reason': reason
                })
                
    df_meta = pd.DataFrame(records)
    df_excl = pd.DataFrame(exclusions)
    
    print("\n--- METADATA EXTRACTION SUMMARY ---")
    print(f"Total valid cohort size: {len(df_meta)}")
    print(f"Total exclusions recorded: {len(df_excl)}")
    print("\nClass distribution in valid cohort:")
    print(df_meta['class_name'].value_counts())
    print("\nBinary target distribution:")
    print(df_meta['target'].value_counts(normalize=True).round(4) * 100)
    print("\nView position distribution:")
    print(df_meta['view_position'].value_counts())
    print("\nBiological sex distribution:")
    print(df_meta['patient_sex'].value_counts())
    print("\nPatient age summary:")
    print(df_meta['patient_age'].describe().round(1))
    
    # Save processed metadata
    os.makedirs(os.path.dirname(OUTPUT_METADATA_CSV), exist_ok=True)
    df_meta.to_csv(OUTPUT_METADATA_CSV, index=False)
    df_excl.to_csv(EXCLUSIONS_LOG_CSV, index=False)
    print(f"\nSaved metadata table to: {OUTPUT_METADATA_CSV}")
    print(f"Saved exclusions log to: {EXCLUSIONS_LOG_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()
