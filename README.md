# [NAACL 2025] Is Translation All You Need? A Study on Solving Multilingual Tasks with Large Language Models

This repository contains the code and datasets for the paper "Is Translation All You Need? A Study on Solving Multilingual Tasks with Large Language Models", which was accepted at NAACL 2025.

## Datasets
All the datasets are in `datasets` folder. The datasets also include the translations of the original data into English, which are used for the experiments in the paper. The datasets include:
- NLP tasks: MGSM, XCOPA, XNLI, XNLI, PAWS-X, MKQA, XL-sum
- Real-world queries: ShareGPT

## Inference
Prepare the environment with yml file
```bash
conda env create -f environment.yml
```
Activate the environment
```bash
conda activate multilingual
```

To run the inference, you can update the parameters and use the following command:
```bash
source 01_run_NLP_tasks.sh
source 02_run_shareGPT.sh
```

## Citation
```bibtex
@misc{liu_is_2024,
	title = {Is {Translation} {All} {You} {Need}? {A} {Study} on {Solving} {Multilingual} {Tasks} with {Large} {Language} {Models}},
	url = {http://arxiv.org/abs/2403.10258},
	publisher = {arXiv},
	author = {Liu, Chaoqun and Zhang, Wenxuan and Zhao, Yiran and Luu, Anh Tuan and Bing, Lidong},
	month = jun,
	year = {2024},
}
```

