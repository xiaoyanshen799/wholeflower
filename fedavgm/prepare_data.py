# prepare_data_fd_npz.py
from pathlib import Path
import numpy as np
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset

DATASET      = "cifar10"
NUM_CLIENTS  = 10
ALPHA        = 0.5
OUT_DIR      = Path("data_partitions1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

partitioner = DirichletPartitioner(
    num_partitions=NUM_CLIENTS,
    partition_by="label",
    alpha=ALPHA,
    min_partition_size=0.0002,
)

fds = FederatedDataset(
    dataset=DATASET,
    partitioners={"train": partitioner, "test": 1},   # "test": 1 ⇒ 不分区
)

# 存 Train 分区
pad = max(5, len(str(NUM_CLIENTS)))
for cid in range(NUM_CLIENTS):
    ds = fds.load_partition(cid, "train").with_format("numpy")
    np.savez_compressed(
        OUT_DIR / f"client_{cid:0{pad}d}.npz",
        x_train=ds["img"],     # CIFAR-10 中字段名是 "img"
        y_train=ds["label"],
    )

# 存 Test 集合（集中评估用）
test_ds = fds.load_split("test").with_format("numpy")
np.savez_compressed(
    OUT_DIR / "test.npz",
    x_test=test_ds["img"],
    y_test=test_ds["label"],
)

print(f"Saved {NUM_CLIENTS} partitions + test set to {OUT_DIR.resolve()}")