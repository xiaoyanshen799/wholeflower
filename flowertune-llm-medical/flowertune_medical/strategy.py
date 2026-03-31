"""flowertune-medical: A Flower / FlowerTune app."""

from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation.

    This class behaves just like FedAvg but also tracks the communication
    costs associated with `train` over FL rounds.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        messages = super().configure_train(server_round, arrays, config, grid)

        # Track communication costs
        self.comm_tracker.track(messages)

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        replies = list(replies)

        # Track communication costs
        self.comm_tracker.track(replies)
        self._log_client_train_metrics(server_round, replies)

        arrays, metrics = super().aggregate_train(server_round, replies)

        return arrays, metrics

    def _log_client_train_metrics(
        self, server_round: int, replies: Iterable[Message]
    ) -> None:
        """Log per-client train metrics on the server side."""
        for reply in replies:
            if not reply.has_content():
                continue

            metrics = reply.content.get("metrics")
            if metrics is None:
                continue

            client_info = reply.content.get("client_info")
            partition_id = (
                client_info.get("partition_id", "unknown")
                if client_info is not None
                else "unknown"
            )
            node_id = reply.metadata.src_node_id
            train_runtime = metrics.get("train_runtime")
            t_local_round_s = metrics.get("t_local_round_s")
            train_loss = metrics.get("train_loss")
            num_examples = metrics.get("num-examples")

            log(
                INFO,
                (
                    "Round %s client partition_id=%s node_id=%s "
                    "train_runtime=%s t_local_round_s=%s "
                    "train_loss=%s num_examples=%s"
                ),
                server_round,
                partition_id,
                node_id,
                f"{float(train_runtime):.4f}" if train_runtime is not None else "n/a",
                f"{float(t_local_round_s):.4f}" if t_local_round_s is not None else "n/a",
                f"{float(train_loss):.6f}" if train_loss is not None else "n/a",
                num_examples if num_examples is not None else "n/a",
            )


class CommunicationTracker:
    """Communication costs tracker over FL rounds."""

    def __init__(self):
        self.curr_comm_cost = 0.0

    def track(self, messages: Iterable[Message]):
        comm_cost = (
            sum(
                record.count_bytes()
                for msg in messages
                if msg.has_content()
                for record in msg.content.array_records.values()
            )
            / 1024**2
        )

        self.curr_comm_cost += comm_cost
        log(
            INFO,
            "Communication budget: used %.2f MB (+%.2f MB this round) / 200,000 MB",
            self.curr_comm_cost,
            comm_cost,
        )

        if self.curr_comm_cost > 2e5:
            log(
                WARN,
                "The accumulated communication cost has exceeded 200,000 MB. "
                "Please consider reducing it if you plan to participate "
                "FlowerTune LLM Leaderboard.",
            )
