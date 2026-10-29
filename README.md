[README.md](https://github.com/user-attachments/files/34130028/README.md)
# LLM-HSGNAS

## Overview

Code for **Large Language Models for Hardware-aware Spectral Graph Neural Architecture Search**.

LLM-HSGNAS uses a large language model to generate spectral graph neural network architectures for edge devices. It combines spectral priors and deployment constraints with evolutionary search, using predictive accuracy, inference latency, throughput, energy consumption, and deployment feedback to refine candidate architectures.

![LLM-HSGNAS framework](images/Frame.png)


## Requirements

- Linux or WSL2 with Bash, `timeout`, and `bc`.
- Python 3.10, PyTorch 2.1.2 with CUDA 11.8, and PyTorch Geometric 2.3.1.
- An LLM API endpoint and access to the target hardware for evaluation.

Install dependencies from the repository root:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
python -m pip install -r LLM_HSGNAS/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118 -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
```

## Running the code

In `LLM_HSGNAS/running_2.sh`, set `BASE_DIR` to the absolute path of `LLM_HSGNAS/EoH-main` and `DATASET="PubMed"`. The script uses 20 search rounds, 12 offspring per round, 30 retained elites, and a 2.5% / 2.5% / 95% train / validation / test split.

Configure the environment and launch from the repository root:

```bash
export LLM_API_ENDPOINT="your-endpoint.com"
export LLM_API_KEY="your-api-key"
export LLM_MODEL="gpt-4.1-mini"
export LLM_HSGNAS_RESULT_DIR="$(pwd)/LLM_HSGNAS/EoH-main/examples/result"
export LLM_HSGNAS_DATA_DIR="$(pwd)/LLM_HSGNAS/EoH-main/examples/GNN_layer_generation/filter/data"

bash LLM_HSGNAS/running_2.sh
```

Use an endpoint hostname without `https://` or `/v1`. The script performs generation, evaluation, and elite selection, saving results under `EoH-main/examples/result/`.

## Results

Semi-supervised **PubMed** results from manuscript Table 3, using a 2.5% / 2.5% / 95% train / validation / test split. Accuracy is in %, latency in ms, throughput in 10⁴ nodes/s, and energy in J per inference.

**NVIDIA RTX 3090**

| Model | Accuracy ↑ | Latency ↓ | Throughput ↑ | Energy ↓ |
| --- | ---: | ---: | ---: | ---: |
| SGC | 81.23 | 3.84 | 512.96 | 1.30 |
| SSGC | 82.05 | 4.40 | 447.90 | 1.50 |
| ChebNet | 83.07 | 2.94 | 705.50 | 0.58 |
| BernNet | 83.12 | 2.55 | 772.49 | 0.34 |
| ARMA | 81.02 | 2.94 | 671.32 | 0.87 |
| ADP-GNN | 81.93 | 3.09 | 638.76 | 0.73 |
| AutoSGNN | 84.09 | 3.96 | 498.19 | 0.70 |
| **LLM-HSGNAS** | 84.81 | 1.79 | 1102.86 | 0.39 |

**Intel Core i9-12900K**

| Model | Accuracy ↑ | Latency ↓ | Throughput ↑ | Energy ↓ |
| --- | ---: | ---: | ---: | ---: |
| SGC | 81.23 | 137.30 | 14.46 | 11.33 |
| SSGC | 82.05 | 134.00 | 14.72 | 12.13 |
| ChebNet | 83.07 | 64.27 | 30.70 | 5.82 |
| BernNet | 83.12 | 77.92 | 25.30 | 6.69 |
| ARMA | 81.02 | 100.49 | 19.62 | 8.67 |
| ADP-GNN | 81.93 | 126.04 | 18.67 | 11.62 |
| AutoSGNN | 84.09 | 56.40 | 34.98 | 5.10 |
| **LLM-HSGNAS** | 84.81 | 38.02 | 51.86 | 3.53 |

**Xilinx Kria KV260**

| Model | Accuracy ↑ | Latency ↓ | Throughput ↑ | Energy ↓ |
| --- | ---: | ---: | ---: | ---: |
| SGC | 81.23 | 8679.51 | 0.23 | 46.61 |
| SSGC | 82.05 | 8771.86 | 0.22 | 47.30 |
| ChebNet | 83.07 | 5852.95 | 0.34 | 32.88 |
| BernNet | 83.12 | 3438.19 | 0.57 | 19.03 |
| ARMA | 81.02 | 4210.60 | 0.47 | 22.89 |
| ADP-GNN | 81.93 | 3903.09 | 0.51 | 22.92 |
| AutoSGNN | 84.09 | 4442.40 | 0.44 | 24.42 |
| **LLM-HSGNAS** | 84.81 | 2604.73 | 0.76 | 14.26 |

**NVIDIA Jetson Orin Nano Super**

| Model | Accuracy ↑ | Latency ↓ | Throughput ↑ | Energy ↓ |
| --- | ---: | ---: | ---: | ---: |
| SGC | 81.23 | 1607.17 | 1.23 | 16.07 |
| SSGC | 82.05 | 1608.84 | 1.23 | 16.09 |
| ChebNet | 83.07 | 1022.01 | 1.95 | 6.56 |
| BernNet | 83.12 | 663.42 | 2.97 | 4.20 |
| ARMA | 81.02 | 814.81 | 2.42 | 8.15 |
| ADP-GNN | 81.93 | 720.30 | 2.74 | 5.12 |
| AutoSGNN | 84.09 | 877.39 | 2.25 | 5.50 |
| **LLM-HSGNAS** | 84.81 | 528.79 | 3.73 | 3.30 |

