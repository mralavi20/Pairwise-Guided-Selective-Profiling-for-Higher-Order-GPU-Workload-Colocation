import os
import time
import json
import csv
from datetime import datetime

import torch
import torch.distributed as dist

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Parse options and initialize distributed training (DDP)
    # ------------------------------------------------------------------
    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp()

    # Metrics / logging options
    metrics_dir = getattr(opt, "metrics_dir", "./logs_metrics")
    run_tag = getattr(opt, "run_tag", "")
    max_train_iters = getattr(opt, "max_train_iters", 0)

    os.makedirs(metrics_dir, exist_ok=True)

    if run_tag:
        tag = run_tag
    else:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_name = f"{opt.name}_{tag}"
    metrics_csv_path = os.path.join(metrics_dir, run_name + "_epoch_metrics.csv")
    runmeta_json_path = os.path.join(metrics_dir, run_name + "_runmeta.json")

    # ------------------------------------------------------------------
    # Distributed / device information
    # ------------------------------------------------------------------
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if len(getattr(opt, "gpu_ids", [])) > 0 and torch.cuda.is_available():
        device_type = "cuda"
        gpu_info = []
        for gid in opt.gpu_ids:
            if gid < 0:
                continue
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
        gpu_info = None

    framework_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_enabled": torch.backends.cudnn.enabled,
    }

    # ------------------------------------------------------------------
    # Create dataset and model
    # ------------------------------------------------------------------
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    if rank == 0:
        print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    # ------------------------------------------------------------------
    # Prepare run-level metrics container
    # ------------------------------------------------------------------
    run_start_time = time.time()
    run_metadata = {
        "run_name": run_name,
        "model": opt.model,
        "name": opt.name,
        "dataroot": opt.dataroot,
        "dataset_mode": opt.dataset_mode,
        "direction": getattr(opt, "direction", None),
        "batch_size": opt.batch_size,
        "n_epochs": opt.n_epochs,
        "n_epochs_decay": opt.n_epochs_decay,
        "lr": opt.lr,
        "beta1": getattr(opt, "beta1", None),
        "ngf": getattr(opt, "ngf", None),
        "ndf": getattr(opt, "ndf", None),
        "gpu_ids": getattr(opt, "gpu_ids", []),
        "device_type": device_type,
        "gpu_info": gpu_info,
        "framework": framework_info,
        "world_size": int(world_size),
        "dataset_size": int(dataset_size),
        "epoch_metrics": [],
    }

    csv_file = None
    csv_writer = None
    if rank == 0:
        csv_file = open(metrics_csv_path, "w", newline="")
        fieldnames = [
            "epoch",
            "epoch_time_sec",
            "epoch_images",
            "epoch_throughput_img_per_sec",
            "total_iters",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        with open(runmeta_json_path, "w") as f:
            json.dump(run_metadata, f, indent=2)

    total_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, total_iters, save_result
                )

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0 and rank == 0:
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

            if max_train_iters > 0 and total_iters >= max_train_iters:
                break

        model.update_learning_rate()

        epoch_time = time.time() - epoch_start_time
        if rank == 0:
            epoch_images = epoch_iter
            epoch_throughput = epoch_images / epoch_time if epoch_time > 0 else 0.0

            epoch_metrics = {
                "epoch": int(epoch),
                "epoch_time_sec": float(epoch_time),
                "epoch_images": int(epoch_images),
                "epoch_throughput_img_per_sec": float(epoch_throughput),
                "total_iters": int(total_iters),
            }
            run_metadata["epoch_metrics"].append(epoch_metrics)

            if csv_writer is not None:
                csv_writer.writerow(epoch_metrics)
                csv_file.flush()

            if epoch % opt.save_epoch_freq == 0:
                print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
                model.save_networks("latest")
                model.save_networks(epoch)

            print(
                f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay}\t Time Taken: {epoch_time:.0f} sec"
            )

        if max_train_iters > 0 and total_iters >= max_train_iters:
            break

    run_total_time = time.time() - run_start_time
    run_metadata["run_total_time_sec"] = float(run_total_time)
    run_metadata["total_iters"] = int(total_iters)
    run_metadata["total_images"] = int(total_iters)
    run_metadata["avg_throughput_img_per_sec"] = (
        total_iters / run_total_time if run_total_time > 0 else 0.0
    )

    if rank == 0:
        with open(runmeta_json_path, "w") as f:
            json.dump(run_metadata, f, indent=2)

        if csv_file is not None:
            csv_file.close()

        print(f"=> epoch metrics written to {metrics_csv_path}")
        print(f"=> run metadata written to {runmeta_json_path}")

    cleanup_ddp()
