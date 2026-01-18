# PLONK Interpretability Study

Understanding how PLONK achieves state-of-the-art visual geolocation through attention analysis and representation probing.

## Overview

This project investigates the interpretability of [PLONK](https://github.com/nicolas-dufour/plonk) (Around the World in 80 Timesteps), a generative geolocation model that achieves SOTA performance on multiple benchmarks.

## Research Questions

1. **What visual features does PLONK attend to when predicting locations?**
2. **How much geographic information is encoded in the frozen StreetCLIP backbone vs learned by the flow matching head?**

## Experiments

### Experiment 1: Attention Analysis
- Visualization of attention maps from CLIP ViT backbone
- Comparison of attention patterns in correct vs incorrect predictions
- Analysis by scene type (landmarks, urban, rural)

### Experiment 2: Linear Probing
- Extraction of 1024-dim StreetCLIP features from 50,000 OSV-5M test images
- Training logistic regression classifiers for geographic prediction
- Hierarchical evaluation: Continent → Country → Region

## Key Findings

**[Results will be added after experiments complete]**

- Linear probe achieves X% country accuracy using frozen StreetCLIP features
- PLONK attends primarily to [features] for accurate predictions
- Geographic information is [highly/moderately/minimally] encoded in the pretrained backbone

## Dataset

- **OSV-5M**: OpenStreetView-5M test set (50,000 images from 154 countries)
- Download instructions: [OSV-5M Dataset](https://huggingface.co/datasets/osv5m/osv5m)

## Requirements
```
python >= 3.10
torch >= 2.0
diff-plonk
scikit-learn
numpy
pandas
matplotlib
seaborn
```

## Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/plonk-interpretability.git
cd plonk-interpretability

# Install dependencies
pip install diff-plonk scikit-learn matplotlib seaborn pandas
```

## Usage

**[Coming soon]**

## Project Structure
```
plonk-interpretability/
├── experiments/
│   ├── 01_linear_probing/
│   └── 02_attention_analysis/
├── notebooks/
├── results/
└── presentation/
```

## Citation

**PLONK Paper:**
```bibtex
@article{dufour2024plonk,
  title={Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation},
  author={Dufour, Nicolas and Picard, David and Kalogeiton, Vicky and Landrieu, Loic},
  journal={CVPR},
  year={2025}
}
```

## Acknowledgments

This project builds upon [PLONK](https://github.com/nicolas-dufour/plonk) by Dufour et al. and uses the [OSV-5M](https://osv5m.github.io/) dataset.

## License

MIT

---

**Work in Progress** - Results and code will be updated as experiments complete.
