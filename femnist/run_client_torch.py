import argparse
import logging
import os
import time
import wave
from pathlib import Path
from typing import Tuple

os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models

SAMPLE_RATE = 16000
FRAME_LENGTH = 400
FRAME_STEP = 160
FFT_LENGTH = 512
MEL_BINS = 32
MEL_LOWER_HZ = 80.0
MEL_UPPER_HZ = 7600.0


def _mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mels = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center == left:
            center = min(center + 1, n_freqs - 1)
        if right == center:
            right = min(right + 1, n_freqs - 1)
        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / float(center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / float(right - center)
    return fb


_MEL_FILTER = torch.from_numpy(
    _mel_filter_bank(
        sample_rate=SAMPLE_RATE,
        n_fft=FFT_LENGTH,
        n_mels=MEL_BINS,
        fmin=MEL_LOWER_HZ,
        fmax=MEL_UPPER_HZ,
    )
)


class SpeechCommandsWaveDataset(Dataset):
    def __init__(
        self,
        files: np.ndarray,
        labels: np.ndarray,
        data_root: Path,
        feature_mode: str = "waveform",
    ) -> None:
        self.files = files
        self.labels = labels.astype(np.int64, copy=False)
        self.data_root = data_root
        self.feature_mode = feature_mode

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _load_wav_16k_mono(path: Path) -> np.ndarray:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sampwidth != 2:
            raise ValueError(f"Unsupported sample width {sampwidth} in {path}")

        pcm = np.frombuffer(raw, dtype=np.int16)
        if channels > 1:
            pcm = pcm.reshape(-1, channels)[:, 0]

        if framerate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz audio, got {framerate} in {path}")

        wav = pcm.astype(np.float32) / 32768.0
        if wav.shape[0] < SAMPLE_RATE:
            wav = np.pad(wav, (0, SAMPLE_RATE - wav.shape[0]))
        else:
            wav = wav[:SAMPLE_RATE]

        return wav

    @staticmethod
    def _to_log_mel(wav: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(wav)
        window = torch.hann_window(FRAME_LENGTH)
        spec = torch.stft(
            x,
            n_fft=FFT_LENGTH,
            hop_length=FRAME_STEP,
            win_length=FRAME_LENGTH,
            window=window,
            return_complex=True,
        ).abs().pow(2.0)  # [F, T]
        mel = torch.matmul(_MEL_FILTER, spec)  # [M, T]
        log_mel = torch.log(mel + 1e-6)
        mean = torch.mean(log_mel)
        std = torch.std(log_mel, unbiased=False)
        log_mel = (log_mel - mean) / (std + 1e-6)
        return log_mel.transpose(0, 1).unsqueeze(0)  # [1, T, M]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rel = self.files[idx]
        wav = self._load_wav_16k_mono(self.data_root / Path(rel))
        if self.feature_mode == "logmel":
            x = self._to_log_mel(wav)
        else:
            x = torch.from_numpy(wav).unsqueeze(0)  # [1, 16000]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class SpeechCNN1D(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def _build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "speech_cnn1d":
        return SpeechCNN1D(num_classes=num_classes)
    if model_name == "resnet18":
        model = tv_models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = tv_models.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported torch model: {model_name}")
    # CIFAR/speech-style stem for small log-mel inputs.
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _feature_mode_for_model(model_name: str) -> str:
    return "logmel" if model_name.startswith("resnet") else "waveform"


def _load_partition(data_dir: Path, cid: int) -> tuple[np.ndarray, np.ndarray]:
    pad = max(5, len(str(cid)))
    file = data_dir / f"client_{cid:0{pad}d}.npz"
    if not file.exists():
        raise FileNotFoundError(f"Partition file not found: {file}")
    with np.load(file, allow_pickle=True) as npz:
        if "files" in npz and "labels" in npz:
            return npz["files"], npz["labels"]
        raise ValueError(f"Expected speech_commands partition with files/labels, got keys: {npz.files}")


def _load_label_map(data_dir: Path) -> list[str]:
    path = data_dir / "label_map.txt"
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _split_train_val(files: np.ndarray, labels: np.ndarray, ratio: float = 0.9):
    split = int(ratio * len(files))
    return files[:split], labels[:split], files[split:], labels[split:]


def get_parameters(model: nn.Module):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Parameter length mismatch: expected {len(keys)}, got {len(parameters)}")
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)


def train_one_round(model, loader, device, epochs, lr):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    running_loss = 0.0

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))

    mean_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return mean_loss, acc


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))

    return running_loss / max(total, 1), correct / max(total, 1)


class TorchSpeechClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float,
        local_epochs_override: int | None = None,
        batch_size_override: int | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.local_epochs_override = local_epochs_override
        self.batch_size_override = batch_size_override

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        fit_start = time.time()
        set_parameters(self.model, parameters)
        server_send_time = config.get("server_send_time")
        config_time = config.get("config_time", config.get("confit_time"))
        server_wait_time = config.get("server_wait_time")
        now = time.time()
        server_to_client_ms = None
        if isinstance(server_send_time, (int, float)):
            server_to_client_ms = max(0.0, (now - float(server_send_time)) * 1000.0)

        local_epochs = self.local_epochs_override if self.local_epochs_override is not None else int(config.get("local_epochs", 1))
        train_start = time.time()
        loss, acc = train_one_round(
            self.model,
            self.train_loader,
            self.device,
            epochs=max(1, local_epochs),
            lr=self.lr,
        )
        train_time = time.time() - train_start
        fit_end = time.time()
        metrics = {
            "train_loss": float(loss),
            "train_acc": float(acc),
            "train_time": float(train_time),
            "edcode": 0.0,
            "client_fit_end_time": float(fit_end),
            "fit_elapsed_s": float(fit_end - fit_start),
        }
        if isinstance(server_to_client_ms, (int, float)):
            metrics["server_to_client_ms"] = float(server_to_client_ms)
        if isinstance(server_wait_time, (int, float)):
            metrics["server_wait_time"] = float(server_wait_time)
        if isinstance(config_time, (int, float)):
            metrics["config_time"] = float(config_time)
        return get_parameters(self.model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.val_loader, self.device)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(acc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flower PyTorch client (speech_commands)")
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--data-dir", default="data_partitions_speech_commands")
    parser.add_argument("--dataset", default="speech_commands", choices=["speech_commands"])
    parser.add_argument("--model", default="resnet34")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", "--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", "--local-batch-size", type=int, default=None)
    parser.add_argument("--uplink-num-bits", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    log_path = data_dir / f"client_{args.cid:05d}.log"
    logging.basicConfig(filename=str(log_path), level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.model not in {"speech_cnn1d", "resnet34", "resnet18"}:
        raise SystemExit(f"Unsupported torch model: {args.model}")

    files, labels = _load_partition(data_dir, args.cid)
    label_map = _load_label_map(data_dir)
    train_files, train_labels, val_files, val_labels = _split_train_val(files, labels, ratio=0.9)

    batch_size = args.batch_size if args.batch_size is not None else 20
    feature_mode = _feature_mode_for_model(args.model)
    train_ds = SpeechCommandsWaveDataset(train_files, train_labels, data_dir, feature_mode=feature_mode)
    val_ds = SpeechCommandsWaveDataset(val_files, val_labels, data_dir, feature_mode=feature_mode)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(label_map)
    model = _build_model(args.model, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info("client=%s device=%s samples(train/val)=%s/%s", args.cid, device, len(train_ds), len(val_ds))

    client = TorchSpeechClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        local_epochs_override=args.epochs,
        batch_size_override=args.batch_size,
    )

    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
