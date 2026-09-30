import argparse
import json
import csv
import math
import time
import os
import urllib.request
import tarfile
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import spacy

# -----------------------------
# Languages and special tokens
# -----------------------------
SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ["<unk>", "<pad>", "<bos>", "<eos>"]

# Global device (set in main)
DEVICE = torch.device("cpu")


# -----------------------------
# Simple vocabulary (no torchtext)
# -----------------------------
class SimpleVocab:
    def __init__(self, stoi, itos, default_index: int):
        self.stoi = stoi
        self.itos = itos
        self.default_index = default_index

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens: List[str]):
        return [self.stoi.get(tok, self.default_index) for tok in tokens]

    def lookup_tokens(self, indices: List[int]):
        res = []
        for idx in indices:
            if 0 <= idx < len(self.itos):
                res.append(self.itos[idx])
            else:
                res.append("<unk>")
        return res


def build_vocab_from_sentences(
    sentences: List[str],
    tokenizer,
    min_freq: int = 1,
) -> SimpleVocab:
    counter = Counter()
    for s in sentences:
        tokens = tokenizer(s)
        counter.update(tokens)

    itos = list(SPECIAL_SYMBOLS)
    stoi = {tok: i for i, tok in enumerate(itos)}

    for token, freq in counter.items():
        if freq < min_freq:
            continue
        if token in stoi:
            continue
        idx = len(itos)
        stoi[token] = idx
        itos.append(token)

    default_index = stoi["<unk>"]
    return SimpleVocab(stoi, itos, default_index)


# -----------------------------
# Dataset wrapper
# -----------------------------
class TranslationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# -----------------------------
# Text transforms
# -----------------------------
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids):
    """
    Add BOS/EOS and convert to 1D tensor.
    """
    return torch.cat(
        (
            torch.tensor([BOS_IDX], dtype=torch.long),
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor([EOS_IDX], dtype=torch.long),
        )
    )


# -----------------------------
# Multi30k download + dataloaders
# -----------------------------
def build_dataloaders(data_root: str, batch_size: int, num_workers: int):
    train_url = (
        "https://raw.githubusercontent.com/neychev/small_DL_repo/master/"
        "datasets/Multi30k/training.tar.gz"
    )
    valid_url = (
        "https://raw.githubusercontent.com/neychev/small_DL_repo/master/"
        "datasets/Multi30k/validation.tar.gz"
    )

    os.makedirs(data_root, exist_ok=True)

    def files_exist(required_files):
        return all(os.path.isfile(os.path.join(data_root, f)) for f in required_files)

    def download_and_extract(url: str, required_files):
        if files_exist(required_files):
            print(f"=> using prepared extracted files: {required_files}")
            return

        filename = os.path.join(data_root, os.path.basename(url))

        if not os.path.exists(filename):
            print(f"=> downloading {url}")
            urllib.request.urlretrieve(url, filename)
        else:
            print(f"=> using cached archive {filename}")

        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=data_root)

        if not files_exist(required_files):
            raise RuntimeError(
                f"Multi30k extraction did not produce the required files: {required_files}"
            )

    download_and_extract(train_url, ["train.de", "train.en"])
    download_and_extract(valid_url, ["val.de", "val.en"])

    def load_split(prefix: str):
        src_path = os.path.join(data_root, f"{prefix}.de")
        tgt_path = os.path.join(data_root, f"{prefix}.en")
        with open(src_path, encoding="utf-8") as fsrc, open(
            tgt_path, encoding="utf-8"
        ) as ftgt:
            src_lines = [l.rstrip("\n") for l in fsrc]
            tgt_lines = [l.rstrip("\n") for l in ftgt]
        assert len(src_lines) == len(tgt_lines)
        return list(zip(src_lines, tgt_lines))

    train_pairs = load_split("train")
    valid_pairs = load_split("val")

    print(f"=> train pairs: {len(train_pairs)}, valid pairs: {len(valid_pairs)}")

    # ----------------- spaCy tokenizers -----------------
    print("=> loading spaCy tokenizers...")
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    def de_tokenizer(text: str):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def en_tokenizer(text: str):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    token_transform = {
        SRC_LANGUAGE: de_tokenizer,
        TGT_LANGUAGE: en_tokenizer,
    }

    # ----------------- build vocabs from training data -----------------
    print("=> building vocabularies from training data...")
    src_sentences = [src for src, _ in train_pairs]
    tgt_sentences = [tgt for _, tgt in train_pairs]

    vocab_transform = {
        SRC_LANGUAGE: build_vocab_from_sentences(
            src_sentences, token_transform[SRC_LANGUAGE], min_freq=1
        ),
        TGT_LANGUAGE: build_vocab_from_sentences(
            tgt_sentences, token_transform[TGT_LANGUAGE], min_freq=1
        ),
    }

    print(f"   SRC vocab size: {len(vocab_transform[SRC_LANGUAGE])}")
    print(f"   TGT vocab size: {len(vocab_transform[TGT_LANGUAGE])}")

    # ----------------- text_transform pipelines -----------------
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln],
            vocab_transform[ln],
            tensor_transform,
        )

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
            tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    train_dataset = TranslationDataset(train_pairs)
    valid_dataset = TranslationDataset(valid_pairs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, vocab_transform


# -----------------------------
# Transformer model
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(
            -torch.arange(0, emb_size, 2) * math.log(10000.0) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ) -> Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask,
        )


# -----------------------------
# Masks
# -----------------------------
def generate_square_subsequent_mask(sz: int) -> Tensor:
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(
        mask == 1, float(0.0)
    )
    return mask


def create_mask(src: Tensor, tgt: Tensor):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
):
    model.train()
    total_loss = 0.0
    total_sentences = 0
    total_tokens = 0
    batches = 0
    start_time = time.time()

    for src, tgt in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]  # remove last token for input

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]  # shift for output
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        batch_sentences = src.size(1)
        batch_tokens = (tgt_out != PAD_IDX).sum().item()

        total_loss += loss.item()
        total_sentences += batch_sentences
        total_tokens += batch_tokens
        batches += 1

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, batches)
    return avg_loss, total_sentences, total_tokens, elapsed


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module
):
    model.eval()
    total_loss = 0.0
    total_sentences = 0
    total_tokens = 0
    batches = 0
    start_time = time.time()

    for src, tgt in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        batch_sentences = src.size(1)
        batch_tokens = (tgt_out != PAD_IDX).sum().item()

        total_loss += loss.item()
        total_sentences += batch_sentences
        total_tokens += batch_tokens
        batches += 1

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, batches)
    return avg_loss, total_sentences, total_tokens, elapsed


# -----------------------------
# Main (CLI + metrics + logging)
# -----------------------------
def main():
    global DEVICE

    parser = argparse.ArgumentParser(
        description="Transformer / Multi30k Translation Workload (de→en)"
    )
    parser.add_argument(
        "--data-cache-dir",
        default=".data_multi30k",
        type=str,
        help="directory to download/extract Multi30k data",
    )
    parser.add_argument(
        "--epochs",
        default=18,
        type=int,
        help="number of total epochs to run (default: 18)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        help="batch size (default: 128)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        help="number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--emb-size",
        default=512,
        type=int,
        help="Transformer embedding dimension (default: 512)",
    )
    parser.add_argument(
        "--nhead",
        default=8,
        type=int,
        help="number of attention heads (default: 8)",
    )
    parser.add_argument(
        "--ffn-hid-dim",
        default=512,
        type=int,
        help="feed-forward hidden dimension (default: 512)",
    )
    parser.add_argument(
        "--num-encoder-layers",
        default=3,
        type=int,
        help="number of encoder layers (default: 3)",
    )
    parser.add_argument(
        "--num-decoder-layers",
        default=3,
        type=int,
        help="number of decoder layers (default: 3)",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed (default: 0)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="disable CUDA even if available",
    )
    parser.add_argument(
        "--log-file",
        default="metrics_transformer_multi30k.json",
        type=str,
        help="path to JSON file to store run metrics",
    )
    parser.add_argument(
        "--csv-file",
        default="metrics_transformer_multi30k.csv",
        type=str,
        help="path to CSV file to store per-epoch metrics",
    )

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {DEVICE}")

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    run_start_time = time.time()

    # ----------------- Data -----------------
    print("=> building dataloaders (download Multi30k if needed)...")
    train_loader, val_loader, vocab_transform = build_dataloaders(
        data_root=args.data_cache_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    src_vocab_size = len(vocab_transform[SRC_LANGUAGE])
    tgt_vocab_size = len(vocab_transform[TGT_LANGUAGE])
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    print(f"   train samples: {train_size}, valid samples: {val_size}")
    print(f"   SRC vocab size: {src_vocab_size}, TGT vocab size: {tgt_vocab_size}")

    # ----------------- Model, loss, optimizer -----------------
    model = Seq2SeqTransformer(
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        emb_size=args.emb_size,
        nhead=args.nhead,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=args.ffn_hid_dim,
        dropout=0.1,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # GPU info / framework meta
    if use_cuda:
        gpu_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(gpu_index)
        gpu_info = {
            "index": int(gpu_index),
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
            "major": int(props.major),
            "minor": int(props.minor),
        }
    else:
        gpu_info = None

    framework_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
    }

    run_metrics = {
        "run_start_time": datetime.now().isoformat(),
        "model": "Seq2SeqTransformer",
        "src_language": SRC_LANGUAGE,
        "tgt_language": TGT_LANGUAGE,
        "src_vocab_size": int(src_vocab_size),
        "tgt_vocab_size": int(tgt_vocab_size),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "optimizer": "Adam",
        "initial_lr": float(args.lr),
        "num_encoder_layers": int(args.num_encoder_layers),
        "num_decoder_layers": int(args.num_decoder_layers),
        "emb_size": int(args.emb_size),
        "nhead": int(args.nhead),
        "ffn_hid_dim": int(args.ffn_hid_dim),
        "device_type": DEVICE.type,
        "gpu_info": gpu_info,
        "framework": framework_info,
        "train_dataset_size": int(train_size),
        "val_dataset_size": int(val_size),
        "epoch_metrics": [],
    }

    best_val_loss = float("inf")

    # ----------------- Training loop -----------------
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_sentences, train_tokens, train_time = train_epoch(
            model, train_loader, optimizer, loss_fn
        )

        val_loss, val_sentences, val_tokens, val_time = evaluate(
            model, val_loader, loss_fn
        )

        epoch_time = time.time() - epoch_start

        train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

        train_sents_per_sec = (
            train_sentences / train_time if train_time > 0 else None
        )
        val_sents_per_sec = val_sentences / val_time if val_time > 0 else None
        train_tokens_per_sec = train_tokens / train_time if train_time > 0 else None
        val_tokens_per_sec = val_tokens / val_time if val_time > 0 else None

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        tsps = train_sents_per_sec if train_sents_per_sec is not None else 0.0
        vsps = val_sents_per_sec if val_sents_per_sec is not None else 0.0

        print(
            f"Epoch: {epoch:03d} | "
            f"Train loss: {train_loss:.4f} (ppl={train_ppl:.2f}) | "
            f"Val loss: {val_loss:.4f} (ppl={val_ppl:.2f}) | "
            f"Train sents/s: {tsps:.1f} | "
            f"Val sents/s: {vsps:.1f} | "
            f"Epoch time: {epoch_time:.2f}s"
        )

        run_metrics["epoch_metrics"].append(
            {
                "epoch": int(epoch),
                "train_time_sec": float(train_time),
                "val_time_sec": float(val_time),
                "epoch_time_sec": float(epoch_time),
                "train_sentences": int(train_sentences),
                "val_sentences": int(val_sentences),
                "train_tokens": int(train_tokens),
                "val_tokens": int(val_tokens),
                "train_sentences_per_sec": float(train_sents_per_sec)
                if train_sents_per_sec is not None
                else None,
                "val_sentences_per_sec": float(val_sents_per_sec)
                if val_sents_per_sec is not None
                else None,
                "train_tokens_per_sec": float(train_tokens_per_sec)
                if train_tokens_per_sec is not None
                else None,
                "val_tokens_per_sec": float(val_tokens_per_sec)
                if val_tokens_per_sec is not None
                else None,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_ppl": float(train_ppl),
                "val_ppl": float(val_ppl),
                "best_val_loss_so_far": float(best_val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

    # ----------------- Finalize metrics -----------------
    run_total_time = time.time() - run_start_time
    run_metrics["run_total_time_sec"] = float(run_total_time)

    if use_cuda:
        run_metrics["gpu_runtime"] = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    # JSON
    with open(args.log_file, "w") as f:
        json.dump(run_metrics, f, indent=2)
    print(f"=> metrics written to {args.log_file}")

    # CSV
    fieldnames = [
        "epoch",
        "model",
        "device_type",
        "gpu_name",
        "batch_size",
        "train_dataset_size",
        "val_dataset_size",
        "train_time_sec",
        "val_time_sec",
        "epoch_time_sec",
        "train_sentences",
        "val_sentences",
        "train_tokens",
        "val_tokens",
        "train_sentences_per_sec",
        "val_sentences_per_sec",
        "train_tokens_per_sec",
        "val_tokens_per_sec",
        "train_loss",
        "val_loss",
        "train_ppl",
        "val_ppl",
        "best_val_loss_so_far",
        "lr",
    ]

    gpu_name = gpu_info["name"] if gpu_info is not None else None

    with open(args.csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for em in run_metrics["epoch_metrics"]:
            row = {
                "epoch": em["epoch"],
                "model": run_metrics["model"],
                "device_type": run_metrics["device_type"],
                "gpu_name": gpu_name,
                "batch_size": run_metrics["batch_size"],
                "train_dataset_size": run_metrics["train_dataset_size"],
                "val_dataset_size": run_metrics["val_dataset_size"],
                "train_time_sec": em["train_time_sec"],
                "val_time_sec": em["val_time_sec"],
                "epoch_time_sec": em["epoch_time_sec"],
                "train_sentences": em["train_sentences"],
                "val_sentences": em["val_sentences"],
                "train_tokens": em["train_tokens"],
                "val_tokens": em["val_tokens"],
                "train_sentences_per_sec": em["train_sentences_per_sec"],
                "val_sentences_per_sec": em["val_sentences_per_sec"],
                "train_tokens_per_sec": em["train_tokens_per_sec"],
                "val_tokens_per_sec": em["val_tokens_per_sec"],
                "train_loss": em["train_loss"],
                "val_loss": em["val_loss"],
                "train_ppl": em["train_ppl"],
                "val_ppl": em["val_ppl"],
                "best_val_loss_so_far": em["best_val_loss_so_far"],
                "lr": em["lr"],
            }
            writer.writerow(row)

    print(f"=> CSV metrics written to {args.csv_file}")


if __name__ == "__main__":
    main()
