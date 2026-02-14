# MMS-VPR  
# MMS-VPRlib  

MMS-VPRlib is a multi-modal street-level dataset and benchmark library for visual place recognition tasks.  
It contains both image sequences and text sequences to facilitate robust visual place recognition research.  

The repository provides modular model implementations, classification scripts, and a unified experimental pipeline for scalable and reproducible benchmarking.

---

## Quick Start

```bash
git clone https://github.com/yiasun/MMS-VPRlib.git
cd MMS-VPRlib
pip install -r requirements.txt
```

---

## Project Structure

```
MMS-VPRlib/
├── README.md
├── requirements.txt
├── run.py
├── sample_data_texts.xlsx
├── models/
├── scripts/
│   └── classification/
└── raw/
```

In our model code, the initial parent folder for image data is named `raw`, and the text dataset file is `sample_data_texts.xlsx`.  
Pretrained language models are downloaded automatically at runtime when required.

---

## Dataset Overview

### Image Data

Images are organized by class directories:

```
raw/
├── Class_A/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Class_B/
│   ├── image1.jpg
│   └── ...
└── ...
```

Default image root used in the code:

```
../raw
```

---

### Text Data

Textual information is provided in:

`sample_data_texts.xlsx`

| Column Name            | Description                         |
|------------------------|-------------------------------------|
| Type                   | Category of the record              |
| Primary Class (Merged) | Merged primary class label          |
| Code                   | Unique identifier code              |
| Location in the Map    | Geographic location reference       |
| Index                  | Numeric index                       |
| List of Store Names    | Comma-separated list of store names |

---

## Supported Model Categories

### Classical Machine Learning

| Model |
|-------|
| Logistic Regression (LR) |
| SVC |
| Random Forest (RF) |
| KNN |
| Gaussian Naïve Bayes (GNB) |
| MLP |

### Deep Visual Encoders

| Model |
|-------|
| ResNet |
| Vision Transformer (ViT) |

### Vision-Language Pretrained Models

| Model |
|-------|
| CLIP |
| BLIP |

### VPR-Specific Architectures

| Model |
|-------|
| BoQ |
| SALAD |
| CosPlace |
| EigenPlaces |
| MixVPR |
| Patch-NetVLAD |
| SFRS |

### Graph-Based and Multimodal Models

| Model |
|-------|
| GCN |
| GAT |
| HGNN |
| ResNet + HGNN |
| R2Former |

---

## Running Experiments

```bash
python run.py --model lr --num_epochs 20
```

Example:

```bash
python run.py --model resnet --batch_size 32 --num_epochs 50
```

---

## Evaluation Metrics

| Metric |
|--------|
| Accuracy |
| Precision |
| Recall |
| F1-score |

---

## Dependencies

| Package         | Minimum Version |
|-----------------|----------------|
| pillow          | 8.0.0         |
| numpy           | 1.19.0        |
| pandas          | 1.1.0         |
| scikit-learn    | 0.24.0        |
| torch           | 1.8.0         |
| torch-geometric | 2.0.0         |
| transformers    | 4.6.0         |
| matplotlib      | 3.3.0         |
| seaborn         | 0.11.0        |

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Contact

rsun155@aucklanduni.ac.nz
