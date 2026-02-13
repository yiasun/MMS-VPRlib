# MMS-VPR
# MMS-VPRlib

MMS-VPRlib is a multi-modal street-level dataset for visual place recognition tasks. It contains both image sequences and text sequences to facilitate robust visual place recognition research.


## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/yiasun/MMS-VPRlib.git
cd MMS-VPRlib
pip install -r requirements.txt
```

## Project Structure
In our model code, the initial parent folder for image data is named raw, and the text dataset file is Sample Data Texts.xlsx.Note that languagemodel.ipynb requires downloading the GPT-2 model at runtime, so it is not recommended to include the model files in the repository—instead, let run_all.py handle downloading them.
```
MMS-VPRlib/
├── README.md               # Project documentation
├── sample_data_texts.xlsx  # Text data in Excel format
├── requirements.txt        # Python dependencies
├── run_all.py              # Script to execute end-to-end pipeline
└── models/
      ├── model.py               # Example notebook for model training and evaluation
      └──...                     # Additional scripts and resources
```

## Dataset Overview

### Image Data

The dataset includes large numbers of JPEG images organized by class directories. Each class folder contains multiple `.jpg` images representing street-level scenes.

### Text Data

Textual information is provided in a single Excel file (`sample_data_texts.xlsx`) with the following columns:

| Column Name            | Description                         |
| ---------------------- | ----------------------------------- |
| Type                   | Category of the record              |
| Primary Class (Merged) | Merged primary class label          |
| Code                   | Unique identifier code              |
| Location in the Map    | Geographic location reference       |
| Index                  | Numeric index                       |
| List of Store Names    | Comma-separated list of store names |

## Model Performance Highlights

The top three performing models in our benchmarks are: 
- **HeteroGNN + ResNet**
- **ResNet-50** 
- **ViT-B/16**  

## Quick Parameter Tuning

All configurable parameters live in `run_all.py`. You can override any default setting on the command line. For example, to train the logistic regression baseline for 20 epochs:

```bash
python run_all.py --model lr --num_epochs 20
## Usage
```
### Testing with Code

To test the results using the provided code, organize your files as follows:

```

    MMS-VPRlib/
    ├── README.md
    ├── sample_data_texts.xlsx
    ├── requirements.txt
    ├── run_all.py
    ├── models/
    │   ├── model.ipynb
    │   └── …
    └── Edge/
        ├── Eh1-1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── …
        ├── Eh1-2/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── …
        └── Eh2-1/
            ├── image1.jpg
            ├── image2.jpg
            └── …

```
The initial folder of the image dataset set in our model is ../raw. The above is a demonstration.

### Running the Full Pipeline

If you want to execute all scripts at once, install the following additional dependencies:

```bash
pip install nbformat>=5.0.0 nbconvert>=6.0.0
```

Then move `run_all.py` into the target folder and run:

```bash
python run_all.py
```

## Dependencies

This project requires Python 3.7 or higher and depends on the following libraries:

| Package         | Minimum Version |
| --------------- | --------------- |
| pillow          | 8.0.0           |
| numpy           | 1.19.0          |
| pandas          | 1.1.0           |
| scikit-learn    | 0.24.0          |
| torch           | 1.8.0           |
| torch-geometric | 2.0.0           |
| transformers    | 4.6.0           |
| matplotlib      | 3.3.0           |
| seaborn         | 0.11.0          |

Install core dependencies with:

```bash
pip install \
  pillow numpy pandas scikit-learn \
  torch torch-geometric transformers \
  matplotlib seaborn
```

For exact version control, use `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Contact

If you have any questions or suggestions, please contact: [rsun155@aucklanduni.ac.nz](mailto:rsun155@aucklanduni.ac.nz)
