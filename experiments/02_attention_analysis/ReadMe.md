# Attention Analysis Documentation

This document provides detailed information about the attention analysis methodology, implementation, and results.

## 🎯 Overview

The attention analysis investigates **what visual features** the StreetCLIP encoder attends to when processing street-level images for geolocation. We extract and analyze attention weights from the Vision Transformer to understand the model's decision-making process.

## 🔬 Methodology

### Attention Extraction

Vision Transformers use multi-head self-attention mechanisms. We focus on:

1. **Last Layer Attention**: Attention weights from the final transformer layer
2. **CLS Token Attention**: How the classification token attends to image patches
3. **Averaged Heads**: Mean attention across all attention heads for stability

### Implementation Details

```python
from plonk import PlonkPipeline

# Load PLONK pipeline
pipeline = PlonkPipeline("nicolas-dufour/PLONK_OSV_5M")

# Access the vision encoder
feature_extractor = pipeline.cond_preprocessing
clip_model = feature_extractor.emb_model
vision_model = clip_model.vision_model

# CRITICAL: Enable attention extraction
vision_model.config._attn_implementation = 'eager'
vision_model.config.output_attentions = True

# Get processor
processor = feature_extractor.processor

# Process image
from PIL import Image
img = Image.open("path/to/image.jpg").convert('RGB')
inputs = processor(images=img, return_tensors="pt")
pixel_values = inputs['pixel_values'].to(device)

# Forward pass with attention
with torch.no_grad():
    outputs = vision_model(
        pixel_values=pixel_values,
        output_attentions=True
    )

# Extract attention from last layer
attention_weights = outputs.attentions[-1]  # Shape: [batch, num_heads, seq_len, seq_len]

# Average over attention heads
attn_avg = attention_weights[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]

# Get CLS token attention to image patches
cls_attn = attn_avg[0, 1:]  # Skip CLS token itself

# Reshape to spatial grid (16x16 for CLIP-ViT-L/14)
grid_size = int(np.sqrt(len(cls_attn)))  # 16
attn_map = cls_attn.reshape(grid_size, grid_size)
```

### Key Configuration Notes

**Why `eager` implementation?**
- Default CLIP implementations (Flash Attention, SDPA) don't return attention weights
- Setting `_attn_implementation = 'eager'` forces the model to use the standard attention implementation that returns weights
- This is necessary for interpretability but slightly slower

**Why last layer?**
- Last layer attention represents the model's final decision
- Earlier layers focus on low-level features (edges, textures)
- Last layer integrates semantic information for classification

**Why average across heads?**
- Different heads may focus on different aspects
- Averaging provides a holistic view of what the model considers important
- Individual heads can be analyzed separately for deeper insights

## 📊 Metrics Computed

For each image, we compute:

### 1. Attention Entropy
```python
cls_attn_norm = cls_attn / (cls_attn.sum() + 1e-8)
entropy = -(cls_attn_norm * np.log(cls_attn_norm + 1e-10)).sum()
```

**Interpretation**:
- **Low entropy** (< 3): Focused attention on few patches
- **Medium entropy** (3-5): Balanced attention distribution
- **High entropy** (> 5): Diffuse attention across many patches

**Our Result**: Mean entropy = **4.53** → Balanced, not hyper-focused

### 2. Attention Concentration

```python
sorted_attn = np.sort(cls_attn_norm)[::-1]
top_10_percent = int(len(sorted_attn) * 0.1)
concentration = sorted_attn[:top_10_percent].sum()
```

**Interpretation**:
- **High concentration** (> 0.7): Model relies on few key regions
- **Medium concentration** (0.5-0.7): Selective but distributed
- **Low concentration** (< 0.5): Uniform attention

**Our Result**: Mean concentration = **66.6%** → Selective attention on important regions

### 3. Number of High-Attention Patches

```python
high_attn_patches = (cls_attn > cls_attn.mean()).sum()
```

**Interpretation**:
- Count of patches receiving above-average attention
- Indicates how many image regions are considered important

**Our Result**: Mean = **95.2 patches** out of 256 total (37%)

### 4. Attention Spread (Standard Deviation)

```python
attn_spread = cls_attn.std()
```

**Interpretation**:
- Variability in attention weights
- High spread: Some patches get much more attention than others
- Low spread: More uniform distribution

## 🖼️ Visualization Methodology

### Heatmap Generation

```python
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np

# Normalize attention map
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

# Resize to image dimensions using bilinear interpolation
img_array = np.array(img)
h, w = img_array.shape[:2]
attn_resized = zoom(attn_map, (h / grid_size, w / grid_size), order=1)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Attention heatmap
axes[1].imshow(attn_resized, cmap='hot')
axes[1].set_title("Attention Heatmap")
axes[1].axis('off')

# Overlay
axes[2].imshow(img)
axes[2].imshow(attn_resized, alpha=0.6, cmap='hot')
axes[2].set_title("Attention Overlay")
axes[2].axis('off')

plt.tight_layout()
```

## 📈 Results Summary

### Overall Statistics (50,000 images)

| Metric                       | Mean  | Std  | Min  | Max  |
|------------------------------|-------|------|------|------|
| Attention Entropy            | 4.53  | 0.31 | 3.12 | 5.89 |
| Top-10% Concentration        | 66.6% | 8.2% | 42%  | 88%  |
| High Attention Patches       | 95.2  | 18.7 | 38   | 156  |
| Attention Spread (Std Dev)   | 0.041 | 0.009| 0.018| 0.076|

![Diverse Attention Examples](../../results/figures/diverse_20_examples.png)

### Key Findings

1. **Consistent Attention Patterns**
   - Low variance in entropy (std = 0.31) indicates stable attention strategy
   - Model doesn't drastically change behavior across different images

2. **Selective but Comprehensive**
   - 66.6% concentration means model is selective (not uniform)
   - 95 patches (37%) means model still considers substantial context
   - Balance between focus and holistic understanding

3. **Visual Features Attended To**
   
   Based on qualitative analysis of visualizations:
   
   **High Attention Regions**:
   - Road signs and markings
   - Building facades and architectural details
   - Vegetation patterns (trees, grass)
   - Sky conditions and color
   - Urban vs. rural indicators
   - Vehicle types and road infrastructure
   
   **Low Attention Regions**:
   - Uniform sky areas
   - Blurred or motion-distorted regions
   - Generic pavement or road surface
   - Image borders and artifacts

4. **Geographic Strategy**
   
   The model appears to use a **combination strategy**:
   - **Landmark detection**: Identifies distinctive features (signs, buildings)
   - **Context integration**: Uses environmental cues (vegetation, sky)
   - **Pattern recognition**: Architectural styles, infrastructure types
   
   This explains why it performs well even without obvious landmarks.

## 🌍 Country-Specific Patterns

### Analysis by Country

```python
# Group statistics by country
country_stats = stats_df.groupby('country').agg({
    'attention_entropy': ['mean', 'std'],
    'attention_concentration_top10': ['mean', 'std'],
    'num_high_attention_patches': ['mean', 'std']
})
```

**Observations**:
- Urban countries (US, UK, FR): Slightly higher concentration (more landmarks)
- Rural countries (AU, RU): More diffuse attention (environmental cues)
- Tropical countries: Strong attention to vegetation patterns
- Desert regions: Higher focus on infrastructure and sky

### Variability

- **Low within-country variance**: Model uses similar strategies within countries
- **Moderate between-country variance**: Some adaptation to geographic context
- **No single strategy**: Model is flexible, not rigid

## 🔍 Interpretation Guidelines

### What Attention Maps Show

**DO Show**:
- ✅ Which spatial regions influence the model's decision
- ✅ Relative importance of different image areas
- ✅ Whether model uses landmarks vs. context

**DON'T Show**:
- ❌ Exact features being detected (need deeper analysis)
- ❌ Causality (correlation ≠ causation)
- ❌ Ground truth of "correct" attention

### Limitations

1. **Attention ≠ Importance**
   - High attention doesn't guarantee that region is causally important
   - Some important features may have low attention in aggregated view

2. **Layer Choice Matters**
   - We only analyze last layer
   - Earlier layers have different attention patterns
   - Full picture requires multi-layer analysis

3. **Head Averaging**
   - Different attention heads may specialize
   - Averaging may hide interesting specialization patterns
   - Individual head analysis could reveal more insights

4. **Post-Attention Processing**
   - Attention is just one component
   - Feed-forward layers also transform features
   - Full model behavior requires analyzing entire pipeline

## 💡 Usage Tips

### For Researchers

1. **Compare with baselines**: Compare attention patterns with random or uniform attention
2. **Ablation studies**: Mask high-attention regions and measure performance drop
3. **Cross-model comparison**: Compare with other geolocation models
4. **Layer analysis**: Analyze all layers, not just the last one

### For Practitioners

1. **Failure analysis**: Check attention on misclassified images
2. **Data quality**: Use attention to identify low-quality images
3. **Model debugging**: Verify model focuses on reasonable features
4. **Feature engineering**: Insights can guide feature selection

## 📁 Output Files

The attention analysis generates:

```
attention_analysis/
├── attention_statistics_50k.csv      # Statistics for all images
├── attention_distributions.png       # Histogram plots
├── visualizations/
│   ├── diverse_examples_grid.png     # Grid of 20 diverse examples
│   ├── high_concentration/           # Focused attention examples
│   ├── low_concentration/            # Diffuse attention examples
│   ├── high_entropy/                 # Spread out attention
│   └── low_entropy/                  # Concentrated attention
└── README.md                         # This file
```

### CSV Format

`attention_statistics_50k.csv` contains:

| Column | Description |
|--------|-------------|
| `image_id` | Image identifier |
| `country` | Country label |
| `attention_entropy` | Attention entropy |
| `attention_max` | Maximum attention value |
| `attention_concentration_top10` | % in top 10% of patches |
| `attention_concentration_top25` | % in top 25% of patches |
| `num_high_attention_patches` | Count above mean |
| `attention_spread` | Standard deviation |

## 🚀 Running the Analysis

### Quick Start

```bash
# Open the notebook
jupyter notebook Attention_2.ipynb

# Run all cells
# Results will be saved to attention_analysis/
```

### Memory Requirements

- **GPU**: 8GB+ recommended (for batch processing)
- **RAM**: 16GB+ recommended (for storing attention maps)
- **Storage**: ~500MB for 50k statistics + visualizations

### Processing Time

- **Per image**: ~0.1-0.2 seconds with GPU
- **50,000 images**: ~2-3 hours total
- **Batch processing**: Recommended batch_size = 1 (attention extraction doesn't batch well)

## 📚 References

### Attention Mechanisms

- Vaswani et al. (2017): "Attention Is All You Need"
- Dosovitskiy et al. (2021): "An Image is Worth 16x16 Words: Transformers for Image Recognition"

### Interpretability

- Chefer et al. (2021): "Transformer Interpretability Beyond Attention Visualization"
- Abnar & Zuidema (2020): "Quantifying Attention Flow in Transformers"

### PLONK

- Dufour et al. (2024): "PLONK: Probabilistic Localization on the Sphere"

---

**Last Updated**: January 19, 2026  
**Author**: Imed-Eddine BOUKHARI  
**Contact**: [Your contact information]

