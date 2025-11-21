import argparse
from pathlib import Path

import numpy as np

from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset


SUPPORTED_DATASETS = {"cifar10": "cifar10", "fmnist": "fashion_mnist"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare federated partitions using flwr_datasets and export as .npz")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), default="cifar10")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration")
    parser.add_argument("--min-partition-size", type=float, default=0.0002)
    parser.add_argument("--output-dir", default="data_partitions", help="Destination folder")
    args = parser.parse_args()

    dataset_name = SUPPORTED_DATASETS[args.dataset]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partitioner = DirichletPartitioner(
        num_partitions=args.num_clients,
        partition_by="label",
        alpha=args.alpha,
        min_partition_size=args.min_partition_size,
    )

    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner, "test": 1})

    pad = max(5, len(str(args.num_clients)))
    for cid in range(args.num_clients):
        ds = fds.load_partition(cid, "train").with_format("numpy")
        np.savez_compressed(
            out_dir / f"client_{cid:0{pad}d}.npz",
            x_train=ds["img"],
            y_train=ds["label"],
        )
        print(f"Saved client {cid} with {len(ds)} samples")

    # Save centralized test set
    test_ds = fds.load_split("test").with_format("numpy")
    np.savez_compressed(out_dir / "test.npz", x_test=test_ds["img"], y_test=test_ds["label"])
    print("Saved test set with", len(test_ds), "samples to", out_dir)


if __name__ == "__main__":
    main() 