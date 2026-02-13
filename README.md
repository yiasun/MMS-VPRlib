# MMS-VPR

# MMS-VPRlib

MMS-VPRlib is a multi-modal street-level dataset and benchmark library
for visual place recognition tasks.\
It contains both image sequences and text sequences to facilitate robust
visual place recognition research.\
The repository provides modular model implementations, classification
scripts, and a unified experimental pipeline for scalable and
reproducible benchmarking.

------------------------------------------------------------------------

## Quick Start

Clone the repository and install dependencies:

git clone https://github.com/yiasun/MMS-VPRlib.git\
cd MMS-VPRlib\
pip install -r requirements.txt

------------------------------------------------------------------------

## Project Structure

MMS-VPRlib/\
├── README.md\
├── requirements.txt\
├── run.py \# Unified experiment launcher\
├── sample_data_texts.xlsx \# Text metadata file\
├── models/ \# Model implementations\
│ ├── resnet.py\
│ ├── vit.py\
│ ├── gnn_models.py\
│ └── ...\
├── scripts/\
│ └── classification/ \# Training & evaluation scripts\
│ ├── train.py\
│ ├── evaluate.py\
│ └── utils.py\
└── raw/ \# Image dataset root (user-provided)

In our model code, the initial parent folder for image data is named
`raw`, and the text dataset file is `sample_data_texts.xlsx`.\
Pretrained language models (such as GPT-2) are downloaded automatically
at runtime when required.

------------------------------------------------------------------------

## Dataset Overview

### Image Data

The dataset includes large numbers of JPEG images organized by class
directories.\
Each class folder contains multiple `.jpg` images representing
street-level scenes.

Example structure:

raw/\
├── Class_A/\
│ ├── image1.jpg\
│ ├── image2.jpg\
│ └── ...\
├── Class_B/\
│ ├── image1.jpg\
│ └── ...\
└── ...

Default image root used in the code:

../raw

------------------------------------------------------------------------

### Text Data

Textual information is provided in a single Excel file
(`sample_data_texts.xlsx`) with the following columns:

  Column Name              Description
  ------------------------ -------------------------------------
  Type                     Category of the record
  Primary Class (Merged)   Merged primary class label
  Code                     Unique identifier code
  Location in the Map      Geographic location reference
  Index                    Numeric index
  List of Store Names      Comma-separated list of store names

This metadata supports multimodal and semantic-enhanced VPR experiments.

------------------------------------------------------------------------

## Supported Model Categories

### Classical Machine Learning

-   Logistic Regression (LR)\
-   SVC\
-   Random Forest (RF)\
-   KNN\
-   Gaussian Naïve Bayes (GNB)\
-   MLP

### Deep Visual Encoders

-   ResNet\
-   Vision Transformer (ViT)

### Vision-Language Pretrained Models

-   CLIP\
-   BLIP

### VPR-Specific Architectures

-   BoQ\
-   SALAD\
-   CosPlace\
-   EigenPlaces\
-   MixVPR\
-   Patch-NetVLAD\
-   SFRS

### Graph-Based and Multimodal Models

-   GCN\
-   GAT\
-   HGNN\
-   ResNet + HGNN\
-   R2Former

------------------------------------------------------------------------

## Running Experiments

Train a specific model:

python run.py --model lr --num_epochs 20

Example:

python run.py --model resnet --batch_size 32 --num_epochs 50

All parameters can be overridden via command-line arguments.

------------------------------------------------------------------------

## Evaluation Metrics

  Metric
  -----------
  Accuracy
  Precision
  Recall
  F1-score

------------------------------------------------------------------------

## Dependencies

This project requires Python 3.8 or higher and depends on the following
libraries:

  Package           Minimum Version
  ----------------- -----------------
  pillow            8.0.0
  numpy             1.19.0
  pandas            1.1.0
  scikit-learn      0.24.0
  torch             1.8.0
  torch-geometric   2.0.0
  transformers      4.6.0
  matplotlib        3.3.0
  seaborn           0.11.0

Install core dependencies with:

pip install\
pillow numpy pandas scikit-learn\
torch torch-geometric transformers\
matplotlib seaborn

For exact version control, use:

pip install -r requirements.txt

------------------------------------------------------------------------

## Reproducibility

-   Fixed random seeds supported\
-   Standardized train/test splits\
-   Unified evaluation pipeline\
-   Modular design for easy model extension

------------------------------------------------------------------------

## Contact

rsun155@aucklanduni.ac.nz
