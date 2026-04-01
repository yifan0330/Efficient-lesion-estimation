# Brain Lesion Project

This repository contains code and experiments for brain lesion analysis using spatial statistical models.

## Project Structure

- `brain_lesion/`: Core brain lesion analysis modules
- `experiment/`: Experimental scripts and analysis code
- `real_data/`: Real brain imaging data processing (excluded from git)
- `simulation_approximate_model/`: Simulation studies with approximate models

## Key Files

- `experiment/run.py`: Main experiment runner script
- `experiment/brain_regression.py`: Brain regression analysis
- `experiment/model.py`: Statistical models for brain lesion analysis

## Requirements

- Python 3.x
- PyTorch
- NumPy, SciPy
- Nibabel (for neuroimaging data)
- Nilearn
- Dask (for distributed computing)

## Usage

Run experiments using the main script:

```bash
python experiment/run.py --model="SpatialBrainLesion" --n_group=2 --run_inference=True
```

See `experiment/run.py` for full command-line options.

## Data

Large data files (*.npz, *.nii.gz) are excluded from version control. You'll need to generate or obtain the required datasets separately.

## Results

Results and figures are generated in the `results/` and `figures/` directories (excluded from git due to size).
