from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import time
import json
import csv
from datetime import datetime

import torch
import torch.multiprocessing as mp

from environment import atari_env
from utils import read_config
from model import A3Clstm
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
# from gym.configuration import undo_logger_setup


parser = argparse.ArgumentParser(description="A3C")
parser.add_argument(
    "-l", "--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)"
)
parser.add_argument(
    "-ec",
    "--entropy-coef",
    type=float,
    default=0.01,
    help="entropy loss coefficient (default: 0.01)",
)
parser.add_argument(
    "-vc",
    "--value-coef",
    type=float,
    default=0.5,
    help="value coefficient (default: 0.5)",
)
parser.add_argument(
    "-g",
    "--gamma",
    type=float,
    default=0.99,
    help="discount factor for rewards (default: 0.99)",
)
parser.add_argument(
    "-t", "--tau", type=float, default=1.00, help="parameter for GAE (default: 1.00)"
)
parser.add_argument(
    "-s", "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "-w",
    "--workers",
    type=int,
    default=32,
    help="how many training processes to use (default: 32)",
)
parser.add_argument(
    "-ns",
    "--num-steps",
    type=int,
    default=20,
    help="number of forward steps in A3C (default: 20)",
)
parser.add_argument(
    "-mel",
    "--max-episode-length",
    type=int,
    default=10000,
    help="maximum length of an episode (default: 10000)",
)
parser.add_argument(
    "-ev",
    "--env",
    default="PongNoFrameskip-v4",
    help="environment to train on (default: PongNoFrameskip-v4)",
)
parser.add_argument(
    "-so",
    "--shared-optimizer",
    default=True,
    help="use an optimizer with shared statistics.",
)
parser.add_argument("-ld", "--load", action="store_true", help="load a trained model")
parser.add_argument(
    "-sm",
    "--save-max",
    action="store_true",
    help="Save model on every test run high score matched or bested",
)
parser.add_argument(
    "-o",
    "--optimizer",
    default="Adam",
    choices=["Adam", "RMSprop"],
    help="optimizer choice of Adam or RMSprop",
)
parser.add_argument(
    "-lmd",
    "--load-model-dir",
    default="trained_models/",
    help="folder to load trained models from",
)
parser.add_argument(
    "-smd",
    "--save-model-dir",
    default="trained_models/",
    help="folder to save trained models",
)
parser.add_argument("-lg", "--log-dir", default="logs/", help="folder to save logs")
parser.add_argument(
    "-gp",
    "--gpu-ids",
    type=int,
    default=[-1],
    nargs="+",
    help="GPUs to use [-1 CPU only] (default: -1)",
)
parser.add_argument(
    "-a", "--amsgrad", action="store_true", help="Adam optimizer amsgrad parameter"
)
parser.add_argument(
    "--skip-rate",
    type=int,
    default=4,
    metavar="SR",
    help="frame skip rate (default: 4)",
)
parser.add_argument(
    "-hs",
    "--hidden-size",
    type=int,
    default=512,
    help="LSTM Cell number of features in the hidden state h",
)
parser.add_argument(
    "-tl",
    "--tensorboard-logger",
    action="store_true",
    help=(
        "Creates tensorboard logger to see graph of model, view model weights and "
        "biases, and monitor test agent reward progress"
    ),
)
parser.add_argument(
    "-evc",
    "--env-config",
    default="config.json",
    help="environment to crop and resize info (default: config.json)",
)
parser.add_argument(
    "-dss",
    "--distributed-step-size",
    type=int,
    default=[],
    nargs="+",
    help=(
        "use different step size among workers by using a list of step sizes to "
        "distribute among workers to use (default: [])"
    ),
)

parser.add_argument(
    "--max-train-episodes",
    type=int,
    default=0,
    help="if >0, maximum number of TRAIN episodes per worker before exiting (default: 0 = unlimited)",
)

parser.add_argument(
    "--max-test-episodes",
    type=int,
    default=0,
    help="if >0, maximum number of TEST episodes before exiting (default: 0 = unlimited)",
)


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.gpu_ids != [-1]:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")

    run_start_time = time.time()
    run_start_iso = datetime.now().isoformat()

    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for k in setup_json.keys():
        if k in args.env:
            env_conf = setup_json[k]

    env = atari_env(args.env, env_conf, args)
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space, args)

    if args.load:
        saved_state = torch.load(
            f"{args.load_model_dir}{args.env}.dat",
            map_location=lambda storage, loc: storage,
        )
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == "RMSprop":
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == "Adam":
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad
            )
        optimizer.share_memory()
    else:
        optimizer = None

    if args.gpu_ids != [-1] and torch.cuda.is_available():
        device_type = "cuda"
        gpu_info = []
        for gid in args.gpu_ids:
            if gid >= 0 and gid < torch.cuda.device_count():
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
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
    }

    try:
        obs_shape = tuple(env.observation_space.shape)
    except AttributeError:
        obs_shape = None

    try:
        num_actions = int(env.action_space.n)
    except AttributeError:
        num_actions = None

    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"a3c_{args.env}_{run_tag}"

    os.makedirs(args.log_dir, exist_ok=True)
    json_path = os.path.join(args.log_dir, run_name + "_runmeta.json")
    csv_path = os.path.join(args.log_dir, run_name + "_runmeta.csv")

    run_metadata = {
        "run_name": run_name,
        "env": args.env,
        "env_config": args.env_config,
        "max_episode_length": args.max_episode_length,
        "gamma": args.gamma,
        "tau": args.tau,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "amsgrad": bool(args.amsgrad),
        "shared_optimizer": bool(args.shared_optimizer),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "num_steps": int(args.num_steps),
        "distributed_step_size": list(args.distributed_step_size),
        "hidden_size": int(args.hidden_size),
        "skip_rate": int(args.skip_rate),
        "gpu_ids": list(args.gpu_ids),
        "device_type": device_type,
        "observation_shape": obs_shape,
        "num_actions": num_actions,
        "log_dir": args.log_dir,
        "save_model_dir": args.save_model_dir,
        "load_model_dir": args.load_model_dir,
        "framework": framework_info,
        "gpu_info": gpu_info,
        "max_train_episodes": int(args.max_train_episodes),
        "max_test_episodes": int(args.max_test_episodes),
    }

    processes = []

    p = mp.Process(target=test, args=(args, shared_model, env_conf))
    p.start()
    processes.append(p)
    time.sleep(0.001)

    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf)
        )
        p.start()
        processes.append(p)
        time.sleep(0.001)

    for p in processes:
        time.sleep(0.001)
        p.join()

    run_end_time = time.time()
    run_total_time = run_end_time - run_start_time

    run_metadata["run_start_time"] = run_start_iso
    run_metadata["run_total_time_sec"] = float(run_total_time)

    with open(json_path, "w") as f_json:
        json.dump(run_metadata, f_json, indent=2)
    print(f"=> run-level metadata written to {json_path}")

    csv_fieldnames = [
        "run_name",
        "env",
        "workers",
        "num_steps",
        "gamma",
        "tau",
        "lr",
        "optimizer",
        "shared_optimizer",
        "seed",
        "hidden_size",
        "max_episode_length",
        "skip_rate",
        "device_type",
        "gpu_ids",
        "gpu_names",
        "run_start_time",
        "run_total_time_sec",
    ]

    if gpu_info is not None:
        gpu_names = "|".join(f"{g['id']}:{g['name']}" for g in gpu_info)
    else:
        gpu_names = None

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "run_name": run_metadata["run_name"],
                "env": run_metadata["env"],
                "workers": run_metadata["workers"],
                "num_steps": run_metadata["num_steps"],
                "gamma": run_metadata["gamma"],
                "tau": run_metadata["tau"],
                "lr": run_metadata["lr"],
                "optimizer": run_metadata["optimizer"],
                "shared_optimizer": run_metadata["shared_optimizer"],
                "seed": run_metadata["seed"],
                "hidden_size": run_metadata["hidden_size"],
                "max_episode_length": run_metadata["max_episode_length"],
                "skip_rate": run_metadata["skip_rate"],
                "device_type": run_metadata["device_type"],
                "gpu_ids": ",".join(str(i) for i in run_metadata["gpu_ids"]),
                "gpu_names": gpu_names,
                "run_start_time": run_metadata["run_start_time"],
                "run_total_time_sec": run_metadata["run_total_time_sec"],
            }
        )

    print(f"=> run-level CSV metadata written to {csv_path}")
