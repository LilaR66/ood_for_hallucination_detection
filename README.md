# Post-Hoc Out-of-Distribution (OOD) Detection for Hallucination Detection in Large Language Models

This repository was developed during a 5-month internship (April–September 2025) at [IRT Saint Exupéry](https://www.irt-saintexupery.com/fr/) (3 Rue Tarfaya, 31400 Toulouse, France). This internship was part of the [MVA Master’s program](https://www.master-mva.com/) (Mathematics, Vision, Learning)  at [ENS Paris-Saclay](https://ens-paris-saclay.fr/) (École Normale Supérieure Paris-Saclay) for the academic year 2024–2025.

All code was written **from scratch**. While the ideas draw from multiple published research papers, no pre-existing code was reused.

A detailed internship report (formatted similarly to a research paper) provides further details on the project context, motivation, work performed, and analysis of the results obtained. The report and presentation files are located in the `submissions/` directory.

## Abstract 

Hallucinations in large language models (LLMs), defined as fluent yet factually incorrect or unsupported
outputs, remain a major concern for the safe and certifiable deployment of AI in critical
domains. We frame hallucination detection as an out-of-distribution (OOD) detection problem, that
is, identifying inputs that deviate from the training distribution. First, we conducted a comprehensive
survey of the state-of-the-art methods in OOD and hallucination detection. Then, we leverage
post-hoc distance-based OOD detectors on internal model representations, as well as logit-based uncertainty
metrics as a baseline. Specifically, we extract hidden states, attention maps, and logit-based
uncertainty signals from LLaMA-2-7B Chat when evaluated on SQuAD 2.0, distinguishing answerable
(in-distribution) from unanswerable (OOD, hallucination-prone) questions. We evaluate several
descriptors with OOD scoring techniques (DeepKNN, Mahalanobis distance, cosine similarity) as
well as linear probes, and systematically compare prompt-only versus response-based representations.
Our results show that unsupervised OOD signals achieve modest discrimination (AUROC $\approx 0.6$),
with prompt-only embedding-based OOD detectors underperforming generation logit-based baselines.
Supervised probes substantially improve linear separability (up to 0.80 accuracy on embedding-based
descriptors) but at the cost of generalization. Layerwise trajectory analyses are explored and yield
little additional benefit.

## Setup 

**Environment Setup**

Create and activate a Conda environment with Python 3.11.13:

```bash
conda create --name oodhallu_env python=3.11.13
conda activate oodhallu_env
```

**Dependencies**

Install dependencies listed in `requirements.txt`. Note that the requirements file contains all packages recorded with `pip freeze`; it is advisable to review it before installation:

```bash
pip install -r requirements.txt
```

## Dataset 

The project uses the SQuAD 2.0 dataset, which is loaded directly in `src/data_reader/squad_loader.py`. Both the main script (`scripts/main.py`) and notebook (`notebooks/part1_extract_descriptors.ipynb`) call this loader. The dataset is cached automatically by Hugging Face in:

```bash
~/.cache/huggingface/datasets/squad_v2
```

(taking about 1.1 GB). There is no need to manually download the dataset before running the project.

## Model

The LLaMA-2-7B model is loaded through `src/model_loader/llama_loader.py`, called similarly in `scripts/main.py` or `notebooks/part1_extract_descriptors.ipynb`. The model is cached locally in:

```bash
~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/
```

(size approximately 13 GB). Prior Hugging Face authentication via `huggingface-cli login` is required to access this model.


## Notes
The first run of `scripts/main.py` or `notebooks/part1_extract_descriptors.ipynb` may take some time to download and cache the model and dataset. Subsequent runs will be faster.


## Project Structure Overview

```bash
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with commands such as `make create_env`, `make activate_env`,
│                         or `make make_public`
├── README.md          <- The top-level README for developers using this project
│
│
├── draft/             <- Draft directory where unfinished code is stored.
│                         This directory should be ignored for the project.
│
├── data/
│   ├── datasets/      <- Datasets generated when running `scripts/main.py` or 
│                         `notebooks/part1_extract_descriptors.ipynb`. 
│                         Initially empty.
│   └── models/        <- Model predictions generated when running `scripts/main.py` or 
│                         `notebooks/part1_extract_descriptors.ipynb`.
│                         Initially empty.
│
├── notebooks/         
│   ├── part1_extract_descriptors.ipynb  
│   │     <- Main notebook #1. Extracts descriptors (computed from hidden states,
│   │        attention maps, and logits) that will later be used for OOD-based 
│   │        hallucination detection. This notebook also serves as a usage example.
│   ├── part2_analyse_results.ipynb
│         <- Main notebook #2. Applies OOD methods, linear probing, anomaly detection
│            to the descriptors extracted in part 1 (either via the notebook or via 
│            `scripts/main.py`). Performs analysis and visualization of the results.           
│
│
├── scripts/
│   ├── main.py        <- Equivalent to `part1_extract_descriptors.ipynb` but fully 
│                         encapsulated and automated in a script that can be run with 
│                         `python scripts/main.py`.
│
├── results/
│   ├── raw/           <- Raw result files (.csv) generated during analysis with 
│                         `notebooks/part2_analyse_results.ipynb`.
│   └── figures/       <- Figures produced by `notebooks/part2_analyse_results.ipynb`.
│
├── requirements.txt   <- Requirements file (from `pip freeze`) for reproducing the 
│                         analysis environment.
│
├── submissions/       <- Contains the internship report and the oral presentation slides.
│
├── setup.py           <- Configuration file for pip installation
│
└── src/               <- Source code for the project
    │
    ├── analysis/          <- Functions for result analysis (evaluation metrics,
    │                         visualization plots, etc.)
    │
    ├── answer_similarity/ <- Functions to measure similarity between the generated 
    │                         answer and the ground-truth answer
    │
    ├── data_reader/       <- Functions for loading and formatting the SQuAD 2.0 dataset 
    │   │                      (`squad_loader.py`) and for loading/saving data 
    │   │                      (`pickle_io.py`).
    │
    ├── inference/         <- Functions to input (context, question) pairs into the model, 
    │   │                      let the model generate an answer, and extract internal 
    │   │                      descriptors (hidden states, attention maps, logits) 
    │   │                      for each transformer layer, either for the prompt alone, 
    │   │                      the generation alone, or the concatenation prompt + generation.
    │
    ├── model_loader/      <- Utilities to load LLaMA-2-7B locally
    │
    ├── ood_methods/       <- Functions to apply OOD detection (DeepKNN, cosine similarity,
    │   │                      Mahalanobis distance) on the internal descriptors extracted 
    │   │                      via `part1_extract_descriptors.ipynb` or `scripts/main.py`.  
    │   │                      Also includes supervised probes (logistic regression) and 
    │   │                      anomaly detection methods (Isolation Forest, One-Class SVM).
    │
    └── utils/             <- General-purpose utility functions

```

## Usage

**Step 1:** Extract internal descriptors from the LLM (hidden states, attention maps, logits) to be used for hallucination detection methods, linear probing or anomaly detection.

- Run the notebook `notebooks/part1_extract_descriptors.ipynb`

or

- Run the script:

```bash
python scripts/main.py
```

**Step 2:** Apply OOD, linear probe methods, and anomaly detection algorithms on extracted descriptors. Analyze and visualize the results using:

- The notebook `notebooks/part2_analyse_results.ipynb`


