"""Validated, versioned scalar messages shared by server and clients."""

import math
from dataclasses import asdict, dataclass, fields
from typing import Mapping

PREFIX = "pacer."


def positive_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    run_id: str
    config_version: int
    theta_target_s: float
    deadline_s: float
    ref_a: float
    q: float
    q_lower: float
    q_upper: float
    schema_version: int = 1
    timing_basis: str = "local_train_wall_s"
    gamma_window: int = 50
    gamma_min_samples: int = 20
    violation_patience: int = 3
    feedback_max_age_rounds: int = 2

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        integer(self.schema_version, "schema_version")
        if self.schema_version != 1:
            raise ValueError("Only pacer schema_version=1 is supported")
        integer(self.config_version, "config_version")
        for name in ("theta_target_s", "deadline_s", "ref_a"):
            object.__setattr__(self, name, positive_number(getattr(self, name), name))
        if not math.isfinite(self.ref_a * self.theta_target_s) or self.ref_a * self.theta_target_s == 0:
            raise ValueError("Reference scale a * theta must be finite and positive")
        for name in ("q", "q_lower", "q_upper"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite probability")
            object.__setattr__(self, name, float(value))
        if not (0 < self.q < 1 and 0 <= self.q_lower <= self.q <= self.q_upper <= 1):
            raise ValueError("Require 0 < q < 1 and 0 <= q_lower <= q <= q_upper <= 1")
        if self.timing_basis != "local_train_wall_s":
            raise ValueError("Only local_train_wall_s is supported; this is not an end-to-end SLO")
        integer(self.gamma_window, "gamma_window", 3)
        integer(self.gamma_min_samples, "gamma_min_samples", 3)
        if self.gamma_min_samples > self.gamma_window:
            raise ValueError("gamma_min_samples cannot exceed gamma_window")
        integer(self.violation_patience, "violation_patience")
        integer(self.feedback_max_age_rounds, "feedback_max_age_rounds", 0)

    def to_wire(self, server_round):
        integer(server_round, "server_round")
        result = {PREFIX + f.name: getattr(self, f.name) for f in fields(Settings)}
        result[PREFIX + "round"] = server_round
        return result

    @classmethod
    def from_wire(cls, config: Mapping):
        try:
            result = cls(**{f.name: config[PREFIX + f.name] for f in fields(cls)})
            server_round = integer(config[PREFIX + "round"], "pacer.round")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid Pacer command: {exc}") from exc
        return result, server_round


@dataclass(frozen=True)
class ControlConfig(Settings):
    required_client_ids: tuple[str, ...] = ()
    effective_round: int = 1

    def __post_init__(self):
        super().__post_init__()
        integer(self.effective_round, "effective_round")
        ids = self.required_client_ids
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("required_client_ids must be a nonempty list of strings")
        if any(not isinstance(cid, str) or not cid.strip() for cid in ids):
            raise ValueError("Every client ID must be a nonempty string")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate required_client_ids")
        object.__setattr__(self, "required_client_ids", tuple(ids))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Pacer control JSON must be an object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(f"Invalid Pacer control fields: {exc}") from exc

    def to_dict(self):
        result = asdict(self)
        result["required_client_ids"] = list(self.required_client_ids)
        return result
