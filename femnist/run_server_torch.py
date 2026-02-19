import argparse
import os
from pathlib import Path

os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
import wave
from torch_strategy import CSVFedAvg

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
        ).abs().pow(2.0)
        mel = torch.matmul(_MEL_FILTER, spec)
        log_mel = torch.log(mel + 1e-6)
        mean = torch.mean(log_mel)
        std = torch.std(log_mel, unbiased=False)
        log_mel = (log_mel - mean) / (std + 1e-6)
        return log_mel.transpose(0, 1).unsqueeze(0)

    def __getitem__(self, idx: int):
        rel = self.files[idx]
        wav = self._load_wav_16k_mono(self.data_root / Path(rel))
        if self.feature_mode == "logmel":
            x = self._to_log_mel(wav)
        else:
            x = torch.from_numpy(wav).unsqueeze(0)
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
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _feature_mode_for_model(model_name: str) -> str:
    return "logmel" if model_name.startswith("resnet") else "waveform"


def get_parameters(model: nn.Module):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)


def evaluate_model(model, loader, device):
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


def _load_test_partition(data_dir: Path):
    test_file = data_dir / "test_speech_commands.npz"
    if not test_file.exists():
        raise FileNotFoundError(f"Test partition not found: {test_file}")
    with np.load(test_file, allow_pickle=True) as npz:
        if "files" in npz and "labels" in npz:
            files = npz["files"]
            labels = npz["labels"]
        else:
            files = npz["x_test"]
            labels = npz["y_test"]
    return files, labels


def _load_num_classes(data_dir: Path) -> int:
    label_map = data_dir / "label_map.txt"
    if not label_map.exists():
        raise FileNotFoundError(f"Label map not found: {label_map}")
    return len([line for line in label_map.read_text(encoding="utf-8").splitlines() if line.strip()])


def make_fit_config(local_epochs: int, batch_size: int):
    def fit_config(server_round: int):
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
        }

    return fit_config


def make_eval_fn(model: nn.Module, loader: DataLoader, device: torch.device):
    def evaluate(server_round: int, parameters, config):
        set_parameters(model, parameters)
        loss, acc = evaluate_model(model, loader, device)
        return float(loss), {"accuracy": float(acc)}

    return evaluate


def main():
    parser = argparse.ArgumentParser(description="Flower server for PyTorch speech_commands")
    parser.add_argument("--address", default="0.0.0.0:8081")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--reporting-fraction", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dataset", default="speech_commands", choices=["speech_commands"])
    parser.add_argument("--model", default="resnet34")
    parser.add_argument("--data-dir-speech", type=Path, default=Path("data_partitions_speech_commands"))
    parser.add_argument("--csv-path", default="logs/comm_times.csv")
    args = parser.parse_args()

    data_dir = args.data_dir_speech
    if args.model not in {"speech_cnn1d", "resnet34", "resnet18"}:
        raise SystemExit(f"Unsupported torch model: {args.model}")

    num_classes = _load_num_classes(data_dir)
    test_files, test_labels = _load_test_partition(data_dir)
    feature_mode = _feature_mode_for_model(args.model)
    test_ds = SpeechCommandsWaveDataset(test_files, test_labels, data_dir, feature_mode=feature_mode)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args.model, num_classes=num_classes).to(device)
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(model))

    strategy = CSVFedAvg(
        csv_log_path=args.csv_path,
        fraction_fit=args.reporting_fraction,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, int(round(args.clients * args.reporting_fraction))),
        min_evaluate_clients=1,
        min_available_clients=args.clients,
        on_fit_config_fn=make_fit_config(args.local_epochs, args.batch_size),
        evaluate_fn=make_eval_fn(model, test_loader, device),
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
