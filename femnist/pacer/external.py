"""Round-boundary configuration loading and independent event logs."""

import json
import logging
import time
from pathlib import Path

from .protocol import ControlConfig


class JsonlSink:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event):
        record = {"timestamp_unix": time.time(), **event}
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        logging.info("[Pacer] %s", json.dumps(record, allow_nan=False, sort_keys=True))


class FileControlSource:
    """Only accept immutable, increasing versions within one fixed run/roster."""

    def __init__(self, path, sink=None):
        self.path = Path(path)
        self.sink = sink or JsonlSink()
        self.current = self._read()
        if self.current.effective_round != 1:
            raise ValueError("The initial configuration must have effective_round=1")
        self.latest = self.current
        self._last_error = None

    def _read(self):
        with self.path.open(encoding="utf-8") as handle:
            return ControlConfig.from_dict(json.load(handle))

    def snapshot_for_round(self, server_round):
        try:
            candidate = self._read()
            if candidate.run_id != self.current.run_id:
                raise ValueError("Changing run_id requires a new run")
            if candidate.required_client_ids != self.current.required_client_ids:
                raise ValueError("Changing the client roster requires a new run")
            if candidate.config_version < self.latest.config_version:
                raise ValueError("Configuration version went backwards")
            if candidate.config_version == self.latest.config_version and candidate != self.latest:
                raise ValueError("Configuration changed without increasing config_version")
            if candidate.config_version > self.latest.config_version:
                self.latest = candidate
            self._last_error = None
        except (OSError, ValueError) as exc:
            message = str(exc)
            if message != self._last_error:
                self.sink.emit({"event": "config_rejected", "reason": message, "server_round": server_round})
                self._last_error = message
        if self.latest.effective_round <= server_round and self.latest != self.current:
            self.current = self.latest
            self.sink.emit({"event": "config_applied", "server_round": server_round, **self.current.to_dict()})
        return self.current
