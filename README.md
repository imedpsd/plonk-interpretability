# PLONK Interpretability: Understanding Geographic Knowledge in Vision Encoders

**Interpreting PLONK: Geographic Knowledge in Vision Encoders**

This repository contains the code and analysis for understanding how PLONK's vision encoder (StreetCLIP) captures and represents geographic knowledge for image-based geolocation tasks.

## 📋 Overview

PLONK is a generative geolocation model that combines a pre-trained vision transformer (StreetCLIP) with flow matching on the Earth's sphere to predict GPS coordinates from street-level images. This research investigates:

1. **How much geographic knowledge is already encoded in the frozen StreetCLIP encoder?**
2. **What visual features does the model attend to for geolocation?**

## 🎯 Research Questions

- Does StreetCLIP already know geography before flow matching training?
- What parts of images does the model focus on for geolocation?
- Does the model rely on obvious landmarks or contextual environmental cues?

## 🔬 Methodology

### 1. Linear Probing Experiments

We evaluate how much geographic knowledge exists in the frozen encoder by training simple linear classifiers (logistic regression) on extracted features.

**Dataset**: OSV-5M Test Set
- 50,000 images sampled from test set
- Geographic labels: Country, Region, City
- 217 countries, ~2,050 regions, 7,292 cities

**Method**:
1. Extract 1024-dim features from frozen StreetCLIP encoder
2. Filter classes: Keep only labels with ≥ 2 samples
3. Train/test split: 80/20 (40k train, 10k test)
4. Train: Logistic Regression
5. Evaluate: Accuracy on held-out test set

### 2. Attention Analysis

We analyze attention patterns to understand what visual features the model uses for geolocation.

**Approach**:
1. Extract attention weights from last layer (50,000 images)
2. Visualize where model "looks" on 16×16 image patches
3. Compute statistics: entropy, concentration, spatial patterns

## 📊 Key Results

### Linear Probing Results

| Level    | Linear Probe | PLONK (10000) | PLONK Paper | Classes |
|----------|--------------|---------------|-------------|---------|
| Country  | **85.0%**    | 78%           | 76.2%       | 217     |
| Region   | **62%**      | 39%           | 44.2%       | 2,050   |
| City     | **8.2%**     | 6%            | 5.4%        | 7,292   |

**Key Findings**:
- The frozen encoder captures substantial geographic knowledge from pre-training
- Linear probe achieves strong accuracy, especially at country and region levels
- Different tasks: classification vs. coordinate regression explain performance differences

### Attention Analysis Results

| Metric                      | Value      |
|-----------------------------|------------|
| Images Analyzed             | 50,000     |
| Mean Entropy                | 4.53       |
| Top-10% Concentration       | 66.6%      |
| High Attention Patches      | 95.2 (avg) |

**Key Findings**:
- **Moderate entropy**: Model doesn't hyper-focus on single features
- **Selective attention**: 67% weight on top 10% of patches
- **Contextual coverage**: Uses 95 patches (37% of image)
- **Diverse visual cues**: Model combines landmarks AND environmental cues (sky, vegetation, architectural patterns)

## 🔍 Main Insights

1. **Strong Pre-trained Encoder Foundation**: The StreetCLIP encoder already contains substantial geographic knowledge before flow matching training

2. **Contextual Understanding**: The model doesn't just focus on obvious landmarks but uses diverse environmental cues including:
   - Sky patterns and atmospheric conditions
   - Vegetation and natural environment
   - Architectural styles and building patterns
   - Road infrastructure and urban planning

3. **Distributed Attention**: Attention is distributed across ~37% of image patches, indicating holistic scene understanding rather than landmark detection

4. **Two-Stage Architecture Validation**: Results validate PLONK's design choice of using a frozen pre-trained encoder with flow matching

## 🏗️ Repository Structure

```
plonk-interpretability/
├── notebooks/              # Jupyter notebooks for experiments
│   ├── linear_probing.ipynb
│   └── attention_analysis.ipynb
├── src/                   # Source code
│   ├── feature_extraction.py
│   ├── linear_probe.py
│   └── attention_viz.py
├── data/                  # Data directory (not included)
├── results/               # Experimental results
│   ├── linear_probing/
│   └── attention_maps/
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision transformers
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
```

### Running Linear Probing

```python
# Extract features from StreetCLIP
python src/feature_extraction.py --dataset osv5m --split test --num_samples 50000

# Train linear probes
python src/linear_probe.py --level country
python src/linear_probe.py --level region
python src/linear_probe.py --level city
```

### Running Attention Analysis

```python
# Extract and analyze attention maps
python src/attention_viz.py --num_images 50000 --output_dir results/attention_maps
```

## 📈 Future Directions

Potential extensions of this research:

1. **Fine-tuned Encoder Comparison**: Replace frozen encoder with fine-tuned version and retrain flow matching to measure performance impact

2. **Layer-wise Analysis**: Analyze attention patterns at different encoder layers to understand hierarchical geographic feature learning

3. **Failure Case Analysis**: Study encoder behavior on failure cases to identify limitations and potential improvements

4. **Geographic Bias Study**: Analyze if attention patterns vary across different geographic regions and cultural contexts

## 📚 Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{boukhari2026plonk_interpretability,
  author = {Boukhari, Imed-Eddine},
  title = {Interpreting PLONK: Geographic Knowledge in Vision Encoders},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/imedpsd/plonk-interpretability}}
}
```

## 📝 Related Work

- **PLONK**: [Original PLONK paper and implementation](https://github.com/lukas-haas/PLONK)
- **StreetCLIP**: Vision-language model trained on street-level imagery
- **OSV-5M**: Large-scale street-view dataset

## 👤 Author

**Imed-Eddine BOUKHARI**
- Université Paris Cité

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PLONK authors for the original model and codebase
- OSV-5M dataset creators
- Université Paris Cité

---

**Date**: January 19, 2026
