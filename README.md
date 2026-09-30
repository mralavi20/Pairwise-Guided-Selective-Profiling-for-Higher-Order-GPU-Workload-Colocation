# Pairwise-Guided Selective Profiling for Higher-Order GPU Workload Colocation

This repository contains the code and experimental results for our paper on
pairwise-guided selective profiling for GPU workload colocation.

The experiments were conducted using computing clusters and GPUs at the
University of Copenhagen. We evaluated five machine learning workloads on
NVIDIA A100, A40, L40S, and H100 GPUs.

## Workloads

The following workloads are included in the experiments:

| ID | Workload | Dataset / Environment |
|---|---|---|
| W1 | ResNet50 | Imagenette |
| W2 | Transformer | Multi30k |
| W3 | A3C | PongNoFrameskip-v4 |
| W4 | CycleGAN | monet2photo |
| W5 | Recoder Autoencoder | MovieLens-20M |

Each workload is located under `code/workloads/` and includes its source code
and a shell script used to run the workload.

## Repository Structure

```text
.
├── code/
│   ├── workloads/
│   │   ├── workload1/
│   │   │   ├── code/
│   │   │   ├── run_resnet_imagenette.sh
│   │   │   └── source.txt
│   │   ├── workload2/
│   │   │   ├── code/
│   │   │   ├── run_transformer_multi30k.sh
│   │   │   └── source.txt
│   │   ├── workload3/
│   │   │   ├── code/
│   │   │   ├── run_a3c_pong.sh
│   │   │   └── source.txt
│   │   ├── workload4/
│   │   │   ├── code/
│   │   │   ├── run_cyclegan_monet.sh
│   │   │   └── source.txt
│   │   └── workload5/
│   │       ├── code/
│   │       ├── run_recoder_ml20m.sh
│   │       └── source.txt
│   ├── run_two_concurrent_workloads.sh
│   └── run_three_concurrent_workloads.sh
│
├── results/
│   ├── single_run_reports.txt
│   ├── single_run_results.csv
│   ├── two_concurrent_run_reports.txt
│   ├── two_concurrent_run_results.csv
│   ├── three_concurrent_run_reports.txt
│   └── three_concurrent_run_results.csv
│
└── README.md
```

## Running the Workloads

The scripts inside each workload directory run the corresponding workload
individually. For example:

```bash
bash code/workloads/workload1/run_resnet_imagenette.sh
```

The following scripts are used to run two-workload and three-workload
colocation experiments:

```bash
bash code/run_two_concurrent_workloads.sh
bash code/run_three_concurrent_workloads.sh
```

The scripts were developed for our experimental cluster environment and may
require changes to paths, datasets, or environment settings before running on
another system.

## Experimental Results

The `results/` directory contains the measurements used in our experimental
analysis.

The CSV files contain the collected measurements for:

- workloads running individually,
- two concurrently running workloads, and
- three concurrently running workloads.

Each configuration was executed three times on each evaluated GPU.

The corresponding `*_reports.txt` files contain human-readable summaries of
the collected measurements.

## Paper

This repository accompanies the paper:

**Pairwise-Guided Selective Profiling for Higher-Order GPU Workload Colocation**

If you use this code or the experimental results, please cite our paper.

```bibtex
@inproceedings{TODO,
  title  = {Pairwise-Guided Selective Profiling for Higher-Order GPU Workload Colocation}
}
```
