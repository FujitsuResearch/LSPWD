# Unsupervised Discovery of Long-Term Spatiotemporal Periodic Workflows in Human Activities

[Project Page](https://sites.google.com/view/periodicworkflow) | [arXiv](https://www.arxiv.org/abs/2511.14945)

## Abstract

Periodic human activities with implicit workflows are common in manufacturing, sports, and daily life. While short-term periodic activities—characterized by simple structures and high-contrast patterns—have been widely studied, long-term periodic workflows with low-contrast patterns remain largely underexplored.

To bridge this gap, we introduce the first benchmark comprising 580 multimodal human activity sequences featuring long-term periodic workflows. The benchmark supports three evaluation tasks aligned with real-world applications: unsupervised periodic workflow detection, task completion tracking, and procedural anomaly detection. We also propose a lightweight, training-free baseline for modeling diverse periodic workflow patterns.

<img src="asserts/motivation.png" width="70%" alt="Abstract Figure">

## Usage

### Dependencies
Ensure you have the following Python packages installed:
- `numpy`
- `scikit-learn`
- `tqdm`
- `matplotlib`
- `scipy`

You can install them using pip:
```bash
pip install numpy scikit-learn tqdm matplotlib scipy
```

### Download Data
Download the dataset from https://huggingface.co/datasets/Fujitsu/LSPWD_Dataset

and add the data/ folder to the root path.

### Estimation
Run the estimation script to perform unsupervised periodic workflow detection on the dataset. This script generates estimation results in the `outputs/` directory.

```bash
python estimation.py
```

### Evaluation
Run the evaluation script to calculate metrics (MAPE, IoU, MAE) based on the estimation results and ground truth.

```bash
python evaluation.py
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{yang2026unsupervised,
  title={Unsupervised Discovery of Long-Term Spatiotemporal Periodic Workflows in Human Activities},
  author={Yang, Fan and Xie, Quanting and Moteki, Atsunori and Masui, Shoichi and Jiang, Shan and Bisk, Yonatan and Neubig, Graham},
  booktitle={WACV},
  year={2026}
}
```
