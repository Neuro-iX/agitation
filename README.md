# Agitation

This repository presents a deep learning-based tool to quantify subject motion in T1-weighted brain MRI.  
The model used for this tool can be trained using our [research code](https://github.com/chbricout/cortical-motion).

## Getting Started

### Installation

You will need an environment with at least **Python 3.11**. Then run:

```bash
pip install agitation
```

Alternatively, you can clone the repository and use:

```bash
python cli.py
```

instead of `agitation`.

### Setup

#### Model

We use a TorchScript version of our best model.  
All model checkpoints and the final TorchScript file are available on [Zenodo](https://zenodo.org/records/15288225).

The model will be downloaded automatically when needed. However, you can also manually download it with:

```bash
agitation manage check
```

The model is stored in your application data directory. You can retrieve the exact location using the `check` command.

To remove all downloaded data:

```bash
agitation manage delete
```

#### MRI Data

Our model was trained on data preprocessed with Clinica's [T1-linear](https://aramislab.paris.inria.fr/clinica/docs/public/dev/Pipelines/T1_Linear/) pipeline.  
While it may work with any T1-weighted MRI, we **strongly recommend** using the same preprocessing pipeline to ensure consistent results.

### Usage

To quantify motion, use the command:

```bash
agitation inference
```

#### Arguments:

- `-d, --dataset`: Path to the root of the dataset. It must be organized according to BIDS or CAPS (Clinica) standards and contain either an `anat` folder or `t1_linear` for CAPS.
- `-f, --file`: Path to a CSV file describing the data to process. The file must contain at least a `data` column specifying the path to each volume. Other columns will be copied to the output CSV.
- `-g, --gpu`: Flag to enable GPU inference.
- `--cuda`: Specify the GPU index to use (defaults to 0).
- `-o, --output`: Path to the output CSV file.

#### Example:

```bash
agitation inference --dataset <path_to_root> -g --output <path_to_output_file>
```
```bash
agitation inference --file <path_to_csv>
```

### Library

The `agitation` package can also be used as a library to include motion estimation in your projects.

#### Downloading the Model

To manually download the model within your code:

```python
from agitation.data_manager import download_model

download_model()
```

#### Dataloader Inference

To run inference on a Dataloader:

```python
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from agitation.inference import estimate_motion_dl
from agitation.processing import LoadVolume

# Example usage
dataset = Dataset(<your_data_as_a_dict>, transform=LoadVolume())
dataloader = DataLoader(dataset)
estimate_motion_dl(dataloader, cuda=0)
```

#### Batch Inference

To perform inference on a single batch:

```python
import torch

from agitation.config import MODEL_PATH
from agitation.processing import SoftLabelToPred

# Dataloading, cropping, and normalization steps

model = torch.jit.load(
    MODEL_PATH,
    map_location="cuda:0"  # If using CUDA
)
converter = SoftLabelToPred()

with torch.inference_mode():
    prediction = model(data).cpu()
    motions = converter(prediction)
```

## Contributing

### Setup

Once the repository is cloned, install the development dependencies with:

```bash
pip install -r dev_requirements.txt
```

### Tests

#### Test Tools

We use:

- `pytest` for unit tests
- `pytest-cov` for coverage reports (targeting 100% test coverage)

Run tests via:

```bash
pytest --cov
```

Other tools:

- `ruff` for linting and formatting (automatically applied via `pre-commit`)
- Additional code quality tools: `ssort`, `pydocstyle`, `mypy`, and `pylint`

#### Test Data

All test data are extracted from MR-ART:

> Nárai, Á., Hermann, P., Auer, T. et al. Movement-related artefacts (MR-ART) dataset of matched motion-corrupted and clean structural MRI brain scans. *Sci Data* 9, 630 (2022). https://doi.org/10.1038/s41597-022-01694-8

### Deployment

Build the package using:

```bash
python -m build
```

Deploy to PyPI with:

```bash
twine upload dist/*
```
