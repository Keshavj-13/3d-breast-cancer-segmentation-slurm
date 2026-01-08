# 3D Breast Cancer Segmentation with Slurm

Reproducible 3D breast cancer segmentation experiments with multi GPU Slurm training.

## Overview

This repository contains code for running and evaluating 3D medical image segmentation models.
The focus is on reproducible training, ablation studies, and scalable execution using Slurm.
All models are reimplementations of published architectures for comparative research purposes.

This repository does not contain any patient data.

## Key Focus Areas

- 3D medical image segmentation for breast cancer datasets
- Multi GPU training using Slurm clusters
- Reproducible experiments and ablation studies
- Training stability and performance benchmarking

## Project Status

Ongoing research collaboration.

Primary contributions in this repository:
- Training pipelines for 3D segmentation models
- Slurm job orchestration for multi GPU execution
- Ablation experiment setup and execution
- Debugging and performance tuning of distributed training

  
## Data Policy

- No medical imaging data is included in this repository
- NIfTI files (.nii, .nii.gz) are explicitly excluded
- Generated images and Slurm output files are ignored
- Users must supply their own datasets locally

See `.gitignore` for full exclusion rules.

## Evaluation

Evaluation scripts compute standard segmentation metrics such as:
- Dice coefficient
- Intersection over Union
- Validation loss curves

Metrics are reported per experiment and logged for comparison.

## Disclaimer

This code is for research and educational purposes only.
It is not intended for clinical or diagnostic use.

No patient data is distributed with this repository.

## Citation

If you use this code in academic work, please cite the original papers corresponding
to the implemented architectures.

This repository itself should not be cited as a primary research contribution.

## Contact

For questions related to execution or reproducibility, open an issue.
For collaboration verification, supervisor contact can be provided on request.
