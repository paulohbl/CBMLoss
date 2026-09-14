import os
import urllib.request
import shutil
import tarfile
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def create_synthetic_leaf_dataset(base_dir="data/synthetic_leaf", num_samples=1000):
    """
    Generates a synthetic dataset of leaf images directly to disk.
    This simulates a real workflow of loading images and a CSV.
    Concepts: 
    1. 'concept_halo' (Yellow ring)
    2. 'concept_lesion' (Brown spot)
    3. 'concept_sporulation' (White dots)
    """
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    
    data = []
    
    np.random.seed(42)
    for i in tqdm(range(num_samples), desc="Generating Synthetic Leaves"):
        img = Image.new('RGB', (128, 128), color=(34, 139, 34)) # Base green leaf
        draw = ImageDraw.Draw(img)
        
        # Random concepts
        halo = np.random.rand() > 0.5
        lesion = np.random.rand() > 0.5
        spores = np.random.rand() > 0.5
        
        if lesion:
            draw.ellipse([40, 40, 80, 80], fill=(139, 69, 19)) # Brown spot
        if halo:
            draw.ellipse([30, 30, 90, 90], outline=(255, 255, 0), width=5) # Yellow ring
        if spores:
            for _ in range(10):
                x, y = np.random.randint(50, 70, 2)
                draw.ellipse([x, y, x+3, y+3], fill=(255, 255, 255)) # White dots
                
        # Disease class logic: 1 if both lesion and halo, else 0
        disease = 1 if (lesion and halo) else 0
        
        img_name = f"leaf_{i:04d}.png"
        img.save(os.path.join(base_dir, "images", img_name))
        
        # 80/20 train val split
        split = "train" if np.random.rand() < 0.8 else "val"
        
        data.append({
            "image_name": img_name,
            "concept_halo": int(halo),
            "concept_lesion": int(lesion),
            "concept_sporulation": int(spores),
            "label": disease,
            "split": split
        })
        
    df = pd.DataFrame(data)
    df[df['split'] == 'train'].to_csv(os.path.join(base_dir, "train.csv"), index=False)
    df[df['split'] == 'val'].to_csv(os.path.join(base_dir, "val.csv"), index=False)
    print(f"Synthetic Leaf Dataset generated at {base_dir}")

def download_and_prepare_cub200(base_dir="data/CUB_200_2011"):
    """
    Downloads the CUB-200-2011 dataset (1.1GB) and pre-processes the attributes 
    into a flat CSV for CBM training.
    """
    csv_train = os.path.join(base_dir, "train.csv")
    csv_val = os.path.join(base_dir, "val.csv")
    
    # If already fully prepared, exit immediately
    if os.path.exists(csv_train) and os.path.exists(csv_val):
        print(f"Dataset CUB-200 already prepared in {base_dir}!")
        return

    os.makedirs(base_dir, exist_ok=True)
    tar_path = os.path.join(base_dir, "CUB_200_2011.tgz")
    
    # Check if user already copied CUB_200_2011.tgz to Google Drive
    drive_candidates = [
        "/content/drive/MyDrive/CBMLoss_Data/CUB_200_2011.tgz",
        "/content/drive/MyDrive/CBMLoss_Checkpoints/CUB_200_2011.tgz",
        "/content/drive/MyDrive/CUB_200_2011.tgz"
    ]
    for d_path in drive_candidates:
        if os.path.exists(d_path) and not os.path.exists(tar_path):
            print(f"Encontrado arquivo no Google Drive: {d_path}. Copiando para {tar_path}...")
            import shutil
            shutil.copy2(d_path, tar_path)
            break

    if not os.path.exists(os.path.join(base_dir, "images.txt")):
        if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1_000_000_000:
            print("Baixando CUB-200-2011 (~1.1 GB com barra de progresso)...")
            url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
            try:
                import requests
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                with requests.get(url, stream=True, headers=headers, timeout=30) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get("content-length", 0))
                    with open(tar_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc="CUB_200_2011.tgz") as pbar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            except Exception as e:
                print(f"Aviso no download via requests ({e}), tentando urllib com User-Agent...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response, open(tar_path, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
        
        print("Extraindo CUB_200_2011.tgz...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(base_dir))
            
    # Process attributes if CSV not ready
        print("Processing CUB-200-2011 attributes into CSV format...")
        
        images = pd.read_csv(os.path.join(base_dir, "images.txt"), sep=' ', names=['img_id', 'image_name'])
        labels = pd.read_csv(os.path.join(base_dir, "image_class_labels.txt"), sep=' ', names=['img_id', 'label'])
        labels['label'] = labels['label'] - 1  # Convert 1-indexed to 0-indexed for PyTorch
        splits = pd.read_csv(os.path.join(base_dir, "train_test_split.txt"), sep=' ', names=['img_id', 'is_train'])
        
        # Load attributes (img_id, attr_id, is_present, certainty, time)
        attrs = pd.read_csv(os.path.join(base_dir, "attributes", "image_attribute_labels.txt"), 
                            sep=' ', names=['img_id', 'attr_id', 'is_present', 'certainty', 'time'],
                            usecols=['img_id', 'attr_id', 'is_present'])
        
        # Pivot attributes to wide format (one column per concept)
        attrs_wide = attrs.pivot(index='img_id', columns='attr_id', values='is_present').fillna(0).astype(int)
        
        # Rename columns to concept_X
        attrs_wide.columns = [f"concept_{c}" for c in attrs_wide.columns]
        attrs_wide = attrs_wide.reset_index()
        
        # Merge all
        df = images.merge(labels, on='img_id').merge(splits, on='img_id').merge(attrs_wide, on='img_id')
        
        # Save to CSV
        df[df['is_train'] == 1].to_csv(csv_train, index=False)
        df[df['is_train'] == 0].to_csv(csv_val, index=False)
        print(f"CUB-200-2011 processing complete! Concept vectors have size {len(attrs_wide.columns)-1}")

if __name__ == "__main__":
    print("1. Generating Synthetic Leaf Dataset on disk...")
    create_synthetic_leaf_dataset()
    
    # print("\n2. Getting CUB-200-2011 Online Dataset...")
    download_and_prepare_cub200()
    print("\nAll datasets ready for training!")
