# import os
# import sys

# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)



import argparse
import os
import time
import json
import csv
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import glog as log

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.metrics import AveragePrecision, Recall, NDCG, RecommenderEvaluator
from recoder.nn import DynamicAutoencoder
from recoder.utils import dataframe_to_csr_matrix
from recoder.recommender import InferenceRecommender


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Workload 5: Recoder DynamicAutoencoder on MovieLens-20M"
    )

    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ml-20m/pro_sg/",
        help="Directory with preprocessed ML-20M CSVs "
             "(train.csv, validation_tr.csv, validation_te.csv, "
             " test_tr.csv, test_te.csv).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/ml-20m/",
        help="Directory to save model checkpoints (if desired).",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="logs_metrics/ml-20m/",
        help="Directory to save run metrics (JSON and CSV).",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional tag for this run (e.g., GPU type, config) used in filenames.",
    )

    # Training hyperparameters (aligned with the reference script)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (Adam).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=2e-5,
        help="Weight decay (L2 regularization) for Adam.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Validation evaluation frequency (in epochs) inside Recoder.train.",
    )
    parser.add_argument(
        "--eval_num_recommendations",
        type=int,
        default=100,
        help="Number of recommendations per user during evaluation.",
    )
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="bce_ns_d_0.0_n_0.5_200",
        help="Checkpoint prefix for Recoder (used if you want to save snapshots).",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Epoch frequency for checkpointing inside Recoder.train.",
    )
    parser.add_argument(
        "--num_data_workers",
        type=int,
        default=-1,
        help="Number of data workers for Recoder. "
             "If -1, uses mp.cpu_count() when CUDA is enabled, otherwise 0.",
    )

    # Device and reproducibility
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Utility: gather GPU / framework info
# ----------------------------------------------------------------------
def get_device_and_gpu_info(use_cuda_flag: bool):
    if use_cuda_flag and torch.cuda.is_available():
        device_type = "cuda"
        num_gpus = torch.cuda.device_count()
        gpu_info = []
        for gid in range(num_gpus):
            props = torch.cuda.get_device_properties(gid)
            gpu_info.append(
                {
                    "id": int(gid),
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "multi_processor_count": int(props.multi_processor_count),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )
    else:
        device_type = "cpu"
        num_gpus = 0
        gpu_info = None

    framework_info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
    }

    return device_type, num_gpus, gpu_info, framework_info


# ----------------------------------------------------------------------
# Main workload
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Device / GPU information
    # ------------------------------------------------------------------
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device_type, num_gpus, gpu_info, framework_info = get_device_and_gpu_info(use_cuda)

    if args.num_data_workers < 0:
        num_data_workers = mp.cpu_count() if use_cuda else 0
    else:
        num_data_workers = args.num_data_workers

    log.info("Device type: %s, num_gpus: %d", device_type, num_gpus)
    if gpu_info is not None and len(gpu_info) > 0:
        log.info("Primary GPU: %s", gpu_info[0]["name"])

    # ------------------------------------------------------------------
    # Prepare run name and metrics paths
    # ------------------------------------------------------------------
    os.makedirs(args.metrics_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    if args.run_tag:
        tag = args.run_tag
    else:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_name = f"recoder_ml20m_ae_{tag}"
    metrics_json_path = os.path.join(args.metrics_dir, run_name + "_runmeta.json")
    metrics_csv_path = os.path.join(args.metrics_dir, run_name + "_runmetrics.csv")

    # ------------------------------------------------------------------
    # Load data (MovieLens-20M preprocessed: pro_sg split)
    # ------------------------------------------------------------------
    data_dir = args.data_dir
    if not data_dir.endswith(os.path.sep):
        data_dir = data_dir + os.path.sep

    log.info("Loading MovieLens-20M preprocessed data from %s", data_dir)

    common_params = {
        "user_col": "uid",
        "item_col": "sid",
        "inter_col": "watched",
    }

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_tr_df = pd.read_csv(os.path.join(data_dir, "validation_tr.csv"))
    val_te_df = pd.read_csv(os.path.join(data_dir, "validation_te.csv"))
    test_tr_df = pd.read_csv(os.path.join(data_dir, "test_tr.csv"))
    test_te_df = pd.read_csv(os.path.join(data_dir, "test_te.csv"))

    train_matrix, item_id_map, _ = dataframe_to_csr_matrix(
        train_df, **common_params
    )

    val_tr_matrix, _, val_user_id_map = dataframe_to_csr_matrix(
        val_tr_df, item_id_map=item_id_map, **common_params
    )
    val_te_matrix, _, _ = dataframe_to_csr_matrix(
        val_te_df,
        item_id_map=item_id_map,
        user_id_map=val_user_id_map,
        **common_params,
    )

    test_tr_matrix, _, test_user_id_map = dataframe_to_csr_matrix(
        test_tr_df,
        item_id_map=item_id_map,
        **common_params,
    )
    test_te_matrix, _, _ = dataframe_to_csr_matrix(
        test_te_df,
        item_id_map=item_id_map,
        user_id_map=test_user_id_map,
        **common_params,
    )


    # Dataset stats
    num_users_train, num_items_train = train_matrix.shape
    num_interactions_train = int(train_matrix.nnz)

    num_users_val, num_items_val = val_tr_matrix.shape
    num_interactions_val_input = int(val_tr_matrix.nnz)
    num_interactions_val_target = int(val_te_matrix.nnz)

    num_users_test, num_items_test = test_tr_matrix.shape
    num_interactions_test_input = int(test_tr_matrix.nnz)
    num_interactions_test_target = int(test_te_matrix.nnz)

    log.info(
        "Train matrix: %d users, %d items, %d interactions",
        num_users_train,
        num_items_train,
        num_interactions_train,
    )
    log.info(
        "Val (input/target): %d users, %d items, %d / %d interactions",
        num_users_val,
        num_items_val,
        num_interactions_val_input,
        num_interactions_val_target,
    )
    log.info(
        "Test (input/target): %d users, %d items, %d / %d interactions",
        num_users_test,
        num_items_test,
        num_interactions_test_input,
        num_interactions_test_target,
    )

    # Wrap matrices into Recoder datasets
    train_dataset = RecommendationDataset(train_matrix)
    val_dataset = RecommendationDataset(val_tr_matrix, val_te_matrix)
    test_dataset = RecommendationDataset(test_tr_matrix, test_te_matrix)

    # ------------------------------------------------------------------
    # Define model and Recoder trainer (Autoencoder, implicit feedback)
    # ------------------------------------------------------------------
    model = DynamicAutoencoder(
        hidden_layers=[200],
        activation_type="tanh",
        noise_prob=0.5,
        sparse=False,  # keep same as reference script (can be True for sparse gradients)
    )

    trainer = Recoder(
        model=model,
        use_cuda=use_cuda,
        optimizer_type="adam",
        loss="logistic",
        user_based=False,
    )

    # Metrics (for validation inside train, and for explicit test evaluation)
    metrics = [
        Recall(k=20, normalize=True),
        Recall(k=50, normalize=True),
        NDCG(k=100),
    ]

    # Checkpoint prefix (used only if you want snapshots; not required for workload)
    model_checkpoint_prefix = os.path.join(
        args.model_dir, args.checkpoint_prefix
    )

    # ------------------------------------------------------------------
    # Prepare run-level metadata (to be filled during and after training)
    # ------------------------------------------------------------------
    run_metadata = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "workload": "recoder_autoencoder_movielens20m",
        "paths": {
            "data_dir": os.path.abspath(args.data_dir),
            "model_dir": os.path.abspath(args.model_dir),
            "metrics_dir": os.path.abspath(args.metrics_dir),
            "metrics_json": os.path.abspath(metrics_json_path),
            "metrics_csv": os.path.abspath(metrics_csv_path),
        },
        "data": {
            "dataset": "MovieLens-20M",
            "split_format": "pro_sg",
            "num_users_train": int(num_users_train),
            "num_items_train": int(num_items_train),
            "num_interactions_train": int(num_interactions_train),
            "num_users_val": int(num_users_val),
            "num_items_val": int(num_items_val),
            "num_interactions_val_input": int(num_interactions_val_input),
            "num_interactions_val_target": int(num_interactions_val_target),
            "num_users_test": int(num_users_test),
            "num_items_test": int(num_items_test),
            "num_interactions_test_input": int(num_interactions_test_input),
            "num_interactions_test_target": int(num_interactions_test_target),
        },
        "model": {
            "type": "DynamicAutoencoder",
            "hidden_layers": [200],
            "activation_type": "tanh",
            "noise_prob": 0.5,
            "sparse_gradients": False,
        },
        "training_config": {
            "batch_size": int(args.batch_size),
            "num_epochs": int(args.num_epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "negative_sampling": True,
            "lr_milestones": [60, 80],
            "num_data_workers": int(num_data_workers),
            "eval_freq": int(args.eval_freq),
            "eval_num_recommendations": int(args.eval_num_recommendations),
            "checkpoint_prefix": args.checkpoint_prefix,
            "checkpoint_freq": int(args.checkpoint_freq),
        },
        "device": {
            "use_cuda": bool(use_cuda),
            "device_type": device_type,
            "num_gpus": int(num_gpus),
            "gpu_info": gpu_info,
        },
        "framework": framework_info,
        "results": {},  # to be filled after training/evaluation
    }

    # Save initial metadata (without timings) so that it exists even if training fails
    with open(metrics_json_path, "w") as f:
        json.dump(run_metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Training: measure total time and throughput
    # ------------------------------------------------------------------
    log.info("Starting training for %d epochs...", args.num_epochs)
    train_start = time.time()

    try:
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            negative_sampling=True,
            lr_milestones=[60, 80],
            num_data_workers=num_data_workers,
            model_checkpoint_prefix=model_checkpoint_prefix,
            checkpoint_freq=args.checkpoint_freq,
            eval_num_recommendations=args.eval_num_recommendations,
            metrics=metrics,
            eval_freq=args.eval_freq,
        )
    except (KeyboardInterrupt, SystemExit):
        log.warning("Interrupted: saving model state to %s", model_checkpoint_prefix)
        trainer.save_state(model_checkpoint_prefix)
        raise

    train_time_sec = time.time() - train_start
    log.info("Training completed in %.2f seconds.", train_time_sec)

    # We define throughput in terms of *positive interactions* processed.
    # Each epoch processes approximately num_interactions_train positives.
    total_positive_interactions = num_interactions_train * args.num_epochs
    train_throughput = (
        float(total_positive_interactions) / train_time_sec
        if train_time_sec > 0.0
        else 0.0
    )

    log.info(
        "Approximate training throughput: %.2f positive interactions / second",
        train_throughput,
    )

    # ------------------------------------------------------------------
    # Evaluation on test set (top-K recommendation metrics)
    # ------------------------------------------------------------------
    log.info("Starting evaluation on test set...")

    recommender = InferenceRecommender(
        model=trainer,
        num_recommendations=args.eval_num_recommendations,
    )
    evaluator = RecommenderEvaluator(recommender, metrics)

    eval_start = time.time()
    metrics_accumulated = evaluator.evaluate(
        test_dataset, batch_size=args.batch_size
    )
    eval_time_sec = time.time() - eval_start

    # Aggregate metrics: take mean over users/batches
    eval_results = {}
    for metric_obj in metrics_accumulated:
        key = str(metric_obj)  # same representation as in reference test.py
        value = float(np.mean(metrics_accumulated[metric_obj]))
        eval_results[key] = value
        log.info("Test %s = %.6f", key, value)

    # Throughput for evaluation: users per second
    eval_users = num_users_test
    eval_throughput_users_per_sec = (
        float(eval_users) / eval_time_sec if eval_time_sec > 0.0 else 0.0
    )
    log.info(
        "Evaluation completed in %.2f seconds (%.2f users / second).",
        eval_time_sec,
        eval_throughput_users_per_sec,
    )

    # ------------------------------------------------------------------
    # Populate results and write final JSON
    # ------------------------------------------------------------------
    run_metadata["results"] = {
        "training": {
            "train_time_sec": float(train_time_sec),
            "total_positive_interactions": int(total_positive_interactions),
            "train_throughput_pos_interactions_per_sec": float(train_throughput),
        },
        "evaluation": {
            "eval_time_sec": float(eval_time_sec),
            "eval_throughput_users_per_sec": float(eval_throughput_users_per_sec),
            "metrics": eval_results,
        },
    }

    with open(metrics_json_path, "w") as f:
        json.dump(run_metadata, f, indent=2)

    log.info("Run metadata written to %s", metrics_json_path)

    # ------------------------------------------------------------------
    # Write single-row CSV summary for easy comparison across GPUs
    # ------------------------------------------------------------------
    # Base fields
    csv_fieldnames = [
        "run_name",
        "device_type",
        "primary_gpu_name",
        "num_gpus",
        "batch_size",
        "num_epochs",
        "num_users_train",
        "num_items_train",
        "num_interactions_train",
        "train_time_sec",
        "train_throughput_pos_interactions_per_sec",
        "eval_time_sec",
        "eval_throughput_users_per_sec",
    ]

    # Add one column per evaluation metric (prefixed for clarity)
    metric_keys_sorted = sorted(eval_results.keys())
    metric_fieldnames = [f"metric_{k}" for k in metric_keys_sorted]
    csv_fieldnames.extend(metric_fieldnames)

    primary_gpu_name = (
        gpu_info[0]["name"] if gpu_info is not None and len(gpu_info) > 0 else "None"
    )

    row = {
        "run_name": run_name,
        "device_type": device_type,
        "primary_gpu_name": primary_gpu_name,
        "num_gpus": int(num_gpus),
        "batch_size": int(args.batch_size),
        "num_epochs": int(args.num_epochs),
        "num_users_train": int(num_users_train),
        "num_items_train": int(num_items_train),
        "num_interactions_train": int(num_interactions_train),
        "train_time_sec": float(train_time_sec),
        "train_throughput_pos_interactions_per_sec": float(train_throughput),
        "eval_time_sec": float(eval_time_sec),
        "eval_throughput_users_per_sec": float(eval_throughput_users_per_sec),
    }

    for k in metric_keys_sorted:
        row[f"metric_{k}"] = float(eval_results[k])

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerow(row)

    log.info("CSV summary written to %s", metrics_csv_path)
    log.info("Workload 5 finished successfully.")


if __name__ == "__main__":
    main()
