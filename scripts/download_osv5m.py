"""
Download OSV-5M test set images for linear probing experiments.

Usage:
    python scripts/download_osv5m.py --output ./data/osv5m --n_images 50000
"""

from huggingface_hub import hf_hub_download
import zipfile
import os
import argparse

def download_osv5m_test(output_dir, n_zips=5):
    """Download OSV-5M test images (5 zip files, ~8.78 GB total)"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading OSV-5M test set to {output_dir}")
    print("Total size: ~8.78 GB (5 zip files)\n")
    
    for i in range(n_zips):
        filename = f"{str(i).zfill(2)}.zip"
        print(f"[{i+1}/{n_zips}] Downloading {filename}...")
        
        zip_path = hf_hub_download(
            repo_id="osv5m/osv5m",
            filename=f"images/test/{filename}",
            repo_type='dataset',
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"    Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(f"{output_dir}/images/test")
        
        print(f"    ✅ Done\n")
    
    print("✅ All images downloaded!")
    
    # Download metadata CSV
    print("Downloading test.csv metadata...")
    hf_hub_download(
        repo_id="osv5m/osv5m",
        filename="test.csv",
        repo_type='dataset',
        local_dir=output_dir
    )
    print("✅ Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./data/osv5m', help='Output directory')
    args = parser.parse_args()
    
    download_osv5m_test(args.output)
```

---

## **3. Notebooks vs Scripts:**

**My recommendation:**
- Keep your **exploration** in Jupyter notebooks (what you're doing now)
- Create **clean Python scripts** from working notebook cells
- In repo: Provide BOTH

**Structure:**
```
notebooks/
  - exploration.ipynb        # Your messy working notebook
  - linear_probing_demo.ipynb  # Clean version for others

experiments/01_linear_probing/
  - extract_features.py      # Standalone script
  - train_probes.py          # Standalone script
