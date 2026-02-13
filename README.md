# MMS-VPRlib

Multi-Modal Street-Level Visual Place Recognition Library

MMS-VPRlib is a unified research library and dataset for fine-grained
street-level visual place recognition (VPR). It supports image-only,
text-enhanced, graph-based, and multimodal models under a consistent
experimental framework.

------------------------------------------------------------------------

## Overview

MMS-VPRlib provides:

-   A fine-grained street-level dataset
-   A unified experimental pipeline
-   Classical machine learning baselines
-   Deep visual encoders
-   Vision-language pretrained models
-   VPR-specific architectures
-   Graph-based multimodal models

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
├── run.py\
├── sample_data_texts.xlsx\
├── models/\
├── scripts/\
│ └── classification/\
└── raw/

------------------------------------------------------------------------

## Dataset Setup

### Image Data

Images must be organized in the following structure:

raw/\
├── Class_A/\
│ ├── img1.jpg\
│ ├── img2.jpg\
│ └── ...\
├── Class_B/\
│ ├── img1.jpg\
│ └── ...\
└── ...

Each folder represents one fine-grained location class.

Default image root: ../raw

------------------------------------------------------------------------

### Text Metadata

Text metadata file:

sample_data_texts.xlsx

Columns:

-   Type\
-   Primary Class (Merged)\
-   Code\
-   Location in the Map\
-   Index\
-   List of Store Names

------------------------------------------------------------------------

## Supported Model Categories

Classical ML: - LR - SVC - RF - KNN - GNB - MLP

Deep Encoders: - ResNet - ViT

Vision-Language: - CLIP - BLIP

VPR-Specific: - BoQ - SALAD - CosPlace - EigenPlaces - MixVPR -
Patch-NetVLAD - SFRS

Graph-Based: - GCN - GAT - HGNN - ResNet+HGNN - R2Former

------------------------------------------------------------------------

## Running Experiments

Example:

python run.py --model lr --num_epochs 20

All parameters can be overridden via command line.

------------------------------------------------------------------------

## Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1-score

------------------------------------------------------------------------

## Dependencies

Python \>= 3.8\
pillow \>= 8.0\
numpy \>= 1.19\
pandas \>= 1.1\
scikit-learn \>= 0.24\
torch \>= 1.8\
torch-geometric \>= 2.0\
transformers \>= 4.6\
matplotlib \>= 3.3

Install:

pip install -r requirements.txt

------------------------------------------------------------------------

## Contact

rsun155@aucklanduni.ac.nz
