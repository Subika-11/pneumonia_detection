import os
import sys
import io
import time
import zipfile
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ZIP_PATH = r"D:\downloads\pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"
SPLITS_CSV = r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\data\processed\splits.csv"
OUTPUT_IMG_DIR = r"D:\rsna_pneumonia\images_224"
IMAGE_SIZE = (224, 224)

def process_and_save_image(zip_ref, entry, study_uid, out_dir):
    """Extracts DICOM, handles photometric interpretation, resizes, and saves as 8-bit PNG."""
    out_path = os.path.join(out_dir, f"{study_uid}.png")
    if os.path.exists(out_path):
        return True
        
    try:
        data = zip_ref.read(entry)
        dcm = pydicom.dcmread(io.BytesIO(data))
        pix = dcm.pixel_array
        
        try:
            pix = apply_voi_lut(pix, dcm)
        except:
            pass
            
        # Handle MONOCHROME1 inversion
        if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            pix = np.amax(pix) - pix
            
        # Normalize to 0-255 uint8
        pix = pix.astype(np.float32)
        p_min, p_max = np.min(pix), np.max(pix)
        if p_max > p_min:
            pix = (pix - p_min) / (p_max - p_min) * 255.0
        else:
            pix = np.zeros_like(pix)
        pix = pix.astype(np.uint8)
        
        # Save as 224x224 grayscale PNG
        img = Image.fromarray(pix).resize(IMAGE_SIZE, Image.BILINEAR)
        img.save(out_path, format="PNG", optimize=False)
        return True
    except Exception as e:
        print(f"Error processing {study_uid}: {e}")
        return False

def main():
    print("=" * 60)
    print("PHASE 3: PRE-CACHING NORMALIZED 224x224 PNGs TO DRIVE D:")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_IMG_DIR}")
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    
    df = pd.read_csv(SPLITS_CSV)
    print(f"Total cohort images to cache: {len(df)}")
    
    # We will process in batches using a thread pool with open zip instances
    # Since ZipFile is not strictly thread-safe with concurrent reads on a single instance,
    # we open one ZipFile per worker or process in chunks.
    entries_and_uids = list(zip(df['zip_entry'], df['study_uid']))
    
    # Check how many are already processed
    existing = set(os.listdir(OUTPUT_IMG_DIR))
    needed = [(entry, uid) for entry, uid in entries_and_uids if f"{uid}.png" not in existing]
    print(f"Already cached: {len(existing)} | Remaining to cache: {len(needed)}")
    
    if not needed:
        print("All images are already cached!")
        return
        
    num_workers = min(8, os.cpu_count() or 4)
    print(f"Using {num_workers} parallel workers...")
    
    def worker_task(batch):
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            results = []
            for entry, uid in batch:
                res = process_and_save_image(z, entry, uid, OUTPUT_IMG_DIR)
                results.append(res)
            return sum(results)
            
    # Chunk the work
    chunk_size = 250
    chunks = [needed[i:i + chunk_size] for i in range(0, len(needed), chunk_size)]
    
    t0 = time.time()
    total_processed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task, chunk) for chunk in chunks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Caching images"):
            total_processed += f.result()
            
    elapsed = time.time() - t0
    print(f"\nCaching complete! Processed {total_processed} images in {elapsed:.1f} seconds ({total_processed/elapsed:.1f} imgs/sec).")
    
    # Verify disk usage
    total_bytes = sum(os.path.getsize(os.path.join(OUTPUT_IMG_DIR, f)) for f in os.listdir(OUTPUT_IMG_DIR))
    print(f"Total cache size on Drive D:: {total_bytes / (1024**2):.1f} MB")
    print("=" * 60)

if __name__ == '__main__':
    main()
