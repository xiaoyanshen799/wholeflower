"""Client-selection baselines (FedCS, TiFL) built on top of QuantizedFedAvgM."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from strategy import QuantizedFedAvgM


@dataclass(frozen=True)
class FedCSResourceProfile:
    """Resource information used by FedCS selection (paper-style)."""

    # Throughputs in bits/second
    b_down_bps: float
    b_up_bps: float
    # Compute capability in samples/second
    f_samples_per_s: float
    # Local sample count used for update time estimation
    n_samples: int


@dataclass
class ClientHistory:
    """Timing history (used by TiFL only; not used by FedCS)."""

    rounds: int = 0
    avg_train_time_s: float = 0.0
    avg_upload_time_ms: float = 0.0
    last_down_ms: float = 0.0
    last_round: int = 0

    def update(
        self,
        train_time_s: Optional[float],
        down_ms: Optional[float],
        upload_ms: Optional[float],
        round_num: int,
        alpha: float = 0.3,
    ) -> None:
        if train_time_s is not None and train_time_s > 0:
            if self.avg_train_time_s <= 0:
                self.avg_train_time_s = float(train_time_s)
            else:
                self.avg_train_time_s = (1 - alpha) * self.avg_train_time_s + alpha * float(train_time_s)
        if upload_ms is not None and upload_ms > 0:
            if self.avg_upload_time_ms <= 0:
                self.avg_upload_time_ms = float(upload_ms)
            else:
                self.avg_upload_time_ms = (1 - alpha) * self.avg_upload_time_ms + alpha * float(upload_ms)
        if down_ms is not None and down_ms > 0:
            self.last_down_ms = float(down_ms)
        self.rounds += 1
        self.last_round = int(round_num)


class _QuantizedConfigureMixin(QuantizedFedAvgM):
    """Shared helpers for sending parameters (with optional downlink quantization)."""

    def _prepare_parameters_to_send(self, parameters: Parameters) -> Tuple[Parameters, str, int]:
        parameters_to_send = parameters
        payload_label = "float32"

        if self.downlink_quantizer is not None:
            float_weights = parameters_to_ndarrays(parameters)
            quantized_weights, _ = self.downlink_quantizer.encode(float_weights)
            parameters_to_send = ndarrays_to_parameters(quantized_weights)
            payload_label = f"int{self.downlink_quant_bits or self.downlink_quantizer.num_bits}"

        payload_bytes = sum(len(tensor) for tensor in parameters_to_send.tensors)
        return parameters_to_send, payload_label, payload_bytes


class _HistoryAwareStrategy(_QuantizedConfigureMixin):
    """Mixin over QuantizedFedAvgM that tracks per-client timing metrics (TiFL)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._client_history: Dict[str, ClientHistory] = {}

    def _record_client_stats(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]]
    ) -> None:
        now = time.time()
        for client_proxy, fit_res in results:
            cid = getattr(client_proxy, "cid", None) or str(id(client_proxy))
            metrics = fit_res.metrics or {}
            train_time_s = _safe_float(metrics.get("train_time"))
            down_ms = _safe_float(metrics.get("server_to_client_ms"))

            # Prefer direct metric if present, else approximate via arrival timestamps.
            upload_ms = _safe_float(metrics.get("client_to_server_ms"))
            if upload_ms is None:
                server_arrival_time = _safe_float(metrics.get("server_arrival_time"))
                client_fit_end_time = _safe_float(metrics.get("client_fit_end_time"))
                if server_arrival_time is not None and client_fit_end_time is not None:
                    upload_ms = max(0.0, (server_arrival_time - client_fit_end_time) * 1000.0)
                elif client_fit_end_time is not None:
                    upload_ms = max(0.0, (now - client_fit_end_time) * 1000.0)

            hist = self._client_history.get(cid)
            if hist is None:
                hist = ClientHistory()
                self._client_history[cid] = hist
            hist.update(
                train_time_s=train_time_s,
                down_ms=down_ms,
                upload_ms=upload_ms,
                round_num=server_round,
            )


class FedCSStrategy(_QuantizedConfigureMixin):
    """FedCS client selection following the paper's estimation model and greedy rule."""

    def __init__(
        self,
        *args,
        # FedCS timing model parameters
        round_deadline_s: float = 180.0,
        t_cs_s: float = 0.0,
        t_agg_s: float = 0.0,
        # Resource request fraction C in the paper (Protocol 2, Step 2)
        resource_request_fraction: float = 1.0,
        # Paper assumes stable conditions; fetch resource profile once and reuse.
        profile_fetch_once: bool = True,
        profile_refresh_s: float = 1.0e9,
        # Optional HTTP probe endpoint URL passed to clients (e.g., http://host:port)
        profile_url: Optional[str] = None,
        profile_bytes: int = 262144,
        # get_properties timeout (seconds)
        properties_timeout_s: float = 60.0,
        # worker threads for concurrent resource requests
        properties_workers: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.round_deadline_s = float(round_deadline_s)
        self.t_cs_s = float(t_cs_s)
        self.t_agg_s = float(t_agg_s)
        self.resource_request_fraction = float(min(max(resource_request_fraction, 0.0), 1.0))
        self.profile_fetch_once = bool(profile_fetch_once)
        # Allow 0 to disable client-side caching when profile_fetch_once is False.
        self.profile_refresh_s = float(max(profile_refresh_s, 0.0))
        self.profile_url = str(profile_url) if profile_url else None
        self.profile_bytes = int(max(profile_bytes, 0))
        self.properties_timeout_s = float(max(properties_timeout_s, 0.1))
        self.properties_workers = int(max(properties_workers, 1))
        # Cache last known profile per server-side cid (may change per round, but caching helps)
        self._profiles: Dict[str, Dict[str, float]] = {}

    # ----------------------------- Selection logic ----------------------------- #
    def _fetch_profile(self, client: ClientProxy, local_epochs: int) -> Optional[Dict[str, float]]:
        """Query client.get_properties to obtain FedCS resource profile."""
        try:
            cfg: Dict[str, Scalar] = {"local_epochs": int(local_epochs)}
            # If we fetch every round, force clients to bypass their own cache.
            cfg["profile_refresh_s"] = 0.0 if not self.profile_fetch_once else float(self.profile_refresh_s)
            if self.profile_url:
                cfg["profile_url"] = self.profile_url
                cfg["profile_bytes"] = int(self.profile_bytes)
            res = client.get_properties(
                GetPropertiesIns(config=cfg),
                timeout=self.properties_timeout_s,
            )
        except Exception as exc:
            logging.warning("[FedCS] get_properties failed for %s: %s", getattr(client, "cid", "?"), exc)
            return None
        props = dict(getattr(res, "properties", {}) or {})
        out: Dict[str, float] = {}
        # Common keys used by baselines / our client implementation.
        # Note: we intentionally do NOT accept t_ud/t_ud_s; update time must be derived
        # from (n_samples, f_samples_per_s, local_epochs) per paper formulation.
        for key in ("b_down_bps", "b_up_bps", "f_samples_per_s", "n_samples", "theta"):
            if key in props:
                try:
                    out[key] = float(props[key])
                except (TypeError, ValueError):
                    continue
        if not out:
            return None
        return out

    def _estimate_times_from_props(
        self, props: Dict[str, float], model_bits: float, local_epochs: int
    ) -> Optional[Tuple[float, float, float]]:
        """Return (t_dist_s, t_ud_s, t_ul_s) from client properties (paper-style)."""
        # Throughput: allow either explicit bps or a single theta (treated as bps for both).
        b_down = props.get("b_down_bps")
        b_up = props.get("b_up_bps")
        theta = props.get("theta")
        if (b_down is None or b_up is None) and theta is not None and theta > 0:
            # Many baseline implementations return a single 'theta' for throughput.
            b_down = b_down or theta
            b_up = b_up or theta
        if b_down is None or b_up is None or b_down <= 0 or b_up <= 0:
            return None

        # Local update: always derive via (n_samples, f_samples_per_s) per paper formulation.
        n_samples = props.get("n_samples")
        f = props.get("f_samples_per_s")
        if n_samples is None or f is None or f <= 0:
            return None
        t_ud_s = float(local_epochs) * float(n_samples) / float(f)

        t_dist_s = model_bits / float(b_down)
        t_ul_s = model_bits / float(b_up)
        return float(t_dist_s), float(t_ud_s), float(t_ul_s)

    def _select_clients(
        self,
        available: Dict[str, ClientProxy],
        target: int,
        min_required: int,
    ) -> List[ClientProxy]:
        """Greedily pack clients using FedCS timing model and greedy rule."""
        if not available:
            return []

        # Paper objective is max |S| s.t. t = T_cs + T^S_x + theta' + T_agg < T_round.
        # Algorithm 3 (Client Selection in Protocol 2) chooses, at each step:
        #   x = argmax_{k in K'} (T^S_k - T^S + t_k^UL + max(0, t_k^UD - theta))
        # then removes x from K' and accepts it if it fits the deadline.
        selected: List[ClientProxy] = []
        remaining: List[Tuple[str, ClientProxy, Dict[str, float]]] = []
        for cid, proxy in available.items():
            props = self._profiles.get(cid)
            if props:
                remaining.append((cid, proxy, props))

        t_s = 0.0  # current distribution time T^S
        theta = 0.0

        while remaining and len(selected) < target:
            best_idx = -1
            best_val = float("-inf")
            best_est = None

            for idx, (cid, proxy, props) in enumerate(remaining):
                est = self._estimate_times_from_props(
                    props, self._current_model_bits, self._current_local_epochs
                )
                if est is None:
                    continue
                t_dist_s, t_ud_s, t_ul_s = est
                t_s_new = max(t_s, t_dist_s)
                val = (t_s_new - t_s) + t_ul_s + max(0.0, t_ud_s - theta)
                if val > best_val:
                    best_val = val
                    best_idx = idx
                    best_est = (proxy, t_s_new, t_ud_s, t_ul_s, t_dist_s)

            if best_idx == -1:
                break

            proxy, t_s_new, t_ud_s, t_ul_s, _t_dist_s = best_est
            # remove x from K' (regardless of acceptance), per Algorithm 3
            remaining.pop(best_idx)

            theta_prime = theta + t_ul_s + max(0.0, t_ud_s - theta)
            t_total = self.t_cs_s + t_s_new + theta_prime + self.t_agg_s

            if t_total < self.round_deadline_s:
                theta = theta_prime
                t_s = t_s_new
                selected.append(proxy)
                logging.info(
                    "[FedCS][Select] accept client=%s val=%.3f T_s=%.3f theta=%.3f total=%.3f (deadline=%.3f)",
                    getattr(proxy, "cid", "?"),
                    best_val,
                    t_s,
                    theta,
                    t_total,
                    self.round_deadline_s,
                )
            else:
                logging.info(
                    "[FedCS][Select] reject client=%s val=%.3f would_total=%.3f (deadline=%.3f)",
                    getattr(proxy, "cid", "?"),
                    best_val,
                    t_total,
                    self.round_deadline_s,
                )

        return selected

    # ----------------------------- Strategy hooks ----------------------------- #
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        parameters_to_send, payload_label, payload_bytes = self._prepare_parameters_to_send(parameters)
        # Model size in bits, used by the FedCS estimation formulas
        self._current_model_bits = float(payload_bytes) * 8.0
        self._current_local_epochs = int(config.get("local_epochs", 1) or 1)
        payload_mb = payload_bytes / 1_000_000
        payload_mib = payload_bytes / (1024**2)

        # In FedCS, the "resource request" asks |K|*C random clients for resource info.
        # We model this by sampling a subset of currently available clients.
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        if not client_manager.wait_for(min_num_clients, timeout=30):
            logging.warning("[Server][FedCS] Round %s: only %s clients available after wait, need %s", server_round, client_manager.num_available(), min_num_clients)
        raw_available = client_manager.all()
        available_all = {getattr(proxy, "cid", cid): proxy for cid, proxy in raw_available.items()}

        requested_n = max(1, int(round(len(available_all) * self.resource_request_fraction)))
        requested_n = min(requested_n, len(available_all))
        requested_keys = list(available_all.keys())
        random.shuffle(requested_keys)
        requested = {cid: available_all[cid] for cid in requested_keys[:requested_n]}

        logging.info(
            "[Server][FedCS] Round %s: resource-request=%s/%s (C=%.2f) model=%.3fMiB epochs=%s deadline=%.1fs",
            server_round,
            len(requested),
            len(available_all),
            self.resource_request_fraction,
            payload_mib,
            self._current_local_epochs,
            self.round_deadline_s,
        )

        # Resource request: query client-side profiles (in parallel).
        to_fetch: Dict[str, ClientProxy] = {}
        for cid, proxy in requested.items():
            if not (self.profile_fetch_once and cid in self._profiles):
                to_fetch[cid] = proxy

        if to_fetch:
            with ThreadPoolExecutor(max_workers=self.properties_workers) as pool:
                fut_to_cid = {
                    pool.submit(self._fetch_profile, proxy, self._current_local_epochs): cid
                    for cid, proxy in to_fetch.items()
                }
                for fut in as_completed(fut_to_cid):
                    cid = fut_to_cid[fut]
                    props = fut.result()
                    if props is None:
                        continue
                    self._profiles[cid] = props

        cached_ready = sum(1 for cid in requested if cid in self._profiles)
        logging.info(
            "[Server][FedCS] Round %s: profiles_ready=%s/%s (fetch_once=%s refresh_s=%s)",
            server_round,
            cached_ready,
            len(requested),
            self.profile_fetch_once,
            self.profile_refresh_s,
        )

        # Paper objective is to maximize |S| (no fixed fraction_fit cap); only C limits the candidate pool size.
        selected_clients = self._select_clients(
            requested,
            target=len(requested),
            min_required=min_num_clients,
        )

        if selected_clients:
            logging.info(
                "[Server][FedCS] Round %s: sending %s payload %.3f MB (%.3f MiB) to %s clients (deadline %.1fs)",
                server_round,
                payload_label,
                payload_mb,
                payload_mib,
                len(selected_clients),
                self.round_deadline_s,
            )

        fit_ins = FitIns(parameters_to_send, config)
        return [(client, fit_ins) for client in selected_clients]

class TiFLStrategy(_HistoryAwareStrategy):
    """Tier-based client selection following TiFL's profiling+tiering+scheduler flow.

    Paper: TiFL: A Tier-based Federated Learning System.
    Scope: only regulates client selection; training/aggregation remains unchanged.
    """

    def __init__(
        self,
        *args,
        # TiFL parameters (hyperparams ignored by user; kept for algorithmic fidelity)
        num_tiers: int = 5,  # m
        sync_rounds: int = 5,  # number of profiling rounds used for tiering
        profile_warmup_rounds: int = 1,  # ignore initial warmup rounds (e.g., TF first-round overhead)
        tmax_s: float = 60.0,  # Tmax, used for profiling timeout/capping
        prob_update_interval: int = 20,  # I
        credits_per_tier: Optional[int] = None,  # Credits_t; default = unlimited
        # Optional periodic re-profiling (paper mentions can be periodic)
        reprofiling_interval_rounds: int = 0,
        **kwargs,
    ) -> None:
        # Backward-compat for older CLI args in `run_server.py` (not used by TiFL paper).
        legacy_retier_interval = kwargs.pop("retier_interval", None)
        _legacy_fast_tier_bias = kwargs.pop("fast_tier_bias", None)
        super().__init__(*args, **kwargs)
        self.num_tiers = max(2, int(num_tiers))
        self.sync_rounds = max(1, int(sync_rounds))
        self.profile_warmup_rounds = max(0, int(profile_warmup_rounds))
        self.tmax_s = float(max(tmax_s, 0.1))
        self.prob_update_interval = max(1, int(prob_update_interval))
        self.reprofiling_interval_rounds = max(0, int(reprofiling_interval_rounds))
        if self.reprofiling_interval_rounds == 0 and legacy_retier_interval is not None:
            # Treat legacy "retier interval" as periodic re-profiling interval.
            self.reprofiling_interval_rounds = max(0, int(legacy_retier_interval))

        # Profiling state: per-client cumulative response latency L_i (sum of sync_rounds rounds).
        self._latency_sum_s: Dict[str, float] = {}
        self._latency_rounds: Dict[str, int] = {}
        self._dropouts: set[str] = set()

        # Tiering state (after profiling)
        self._tier_of_cid: Dict[str, int] = {}
        self._tiers: List[List[str]] = []  # tier -> list of cids
        self._tier_latency_avg_s: List[float] = []

        # Scheduler state (Algorithm 2)
        self._tier_probs: List[float] = []
        self._tier_credits: List[int] = []
        self._tier_accuracy_by_round: Dict[int, List[Optional[float]]] = {}
        self._current_tier: int = 0  # tier selected in the most recent training round

        self._credits_per_tier = credits_per_tier  # None => effectively unlimited

    # ----------------------------- Profiling/Tiering ----------------------------- #
    def _is_profiling_round(self, server_round: int) -> bool:
        return server_round <= (self.profile_warmup_rounds + self.sync_rounds)

    def _is_profile_warmup_round(self, server_round: int) -> bool:
        return server_round <= self.profile_warmup_rounds

    def _observe_latency_s(self, cid: str, observed_s: Optional[float], *, server_round: int) -> None:
        """Update L_i using TiFL profiling rule with Tmax capping."""
        if self._is_profile_warmup_round(server_round):
            return
        add_s = self.tmax_s
        if observed_s is not None and observed_s > 0:
            add_s = observed_s if observed_s <= self.tmax_s else self.tmax_s
        self._latency_sum_s[cid] = float(self._latency_sum_s.get(cid, 0.0) + add_s)
        self._latency_rounds[cid] = int(self._latency_rounds.get(cid, 0) + 1)

    def _build_tiers(self, available: Dict[str, ClientProxy]) -> None:
        """Build m tiers by splitting the latency histogram into m bins (paper)."""
        # Filter to clients we have profiled and which are not dropouts.
        profiled: List[Tuple[str, float]] = []
        for cid in available.keys():
            if cid in self._dropouts:
                continue
            if cid in self._latency_sum_s and self._latency_rounds.get(cid, 0) > 0:
                # Use total profiling latency L_i for tiering (paper: histogram over collected latencies).
                profiled.append((cid, float(self._latency_sum_s[cid])))

        # Mark dropouts: Li >= sync_rounds * Tmax (note warmup rounds are ignored by design)
        for cid in list(available.keys()):
            if cid in self._dropouts:
                continue
            li_sum = float(self._latency_sum_s.get(cid, 0.0))
            if li_sum >= float(self.sync_rounds) * float(self.tmax_s):
                self._dropouts.add(cid)
        profiled = [(cid, li_sum) for cid, li_sum in profiled if cid not in self._dropouts]

        self._tier_of_cid = {}
        self._tiers = [[] for _ in range(self.num_tiers)]
        self._tier_latency_avg_s = [0.0 for _ in range(self.num_tiers)]

        if not profiled:
            logging.warning("[TiFL] No profiled non-dropout clients; tiers remain empty.")
            return

        values = [li_sum for _, li_sum in profiled]
        vmin = min(values)
        vmax = max(values)
        if vmax <= vmin:
            # All equal; place everyone into the fastest tier.
            for cid, _li_sum in profiled:
                self._tier_of_cid[cid] = 0
                self._tiers[0].append(cid)
        else:
            # Histogram binning into m groups.
            width = (vmax - vmin) / float(self.num_tiers)
            for cid, li_sum in profiled:
                idx = int((li_sum - vmin) / width) if width > 0 else 0
                idx = min(max(idx, 0), self.num_tiers - 1)
                self._tier_of_cid[cid] = idx
                self._tiers[idx].append(cid)

        # Compute average total profiling latency per tier.
        for t in range(self.num_tiers):
            cids = self._tiers[t]
            if not cids:
                self._tier_latency_avg_s[t] = 0.0
                continue
            self._tier_latency_avg_s[t] = float(
                sum(float(self._latency_sum_s.get(cid, 0.0)) for cid in cids) / float(len(cids))
            )

        # Initialize scheduler state
        self._tier_probs = [1.0 / float(self.num_tiers) for _ in range(self.num_tiers)]
        if self._credits_per_tier is None:
            # "Unlimited" credits: large enough so it won't bind in typical runs.
            self._tier_credits = [10**9 for _ in range(self.num_tiers)]
        else:
            self._tier_credits = [max(0, int(self._credits_per_tier)) for _ in range(self.num_tiers)]

        logging.info(
            "[TiFL] Built %s tiers from %s clients (dropouts=%s). Tier sizes=%s tier_avg_latency_sum_s=%s",
            self.num_tiers,
            len(profiled),
            len(self._dropouts),
            [len(t) for t in self._tiers],
            [f"{x:.3f}" for x in self._tier_latency_avg_s],
        )
        for t in range(self.num_tiers):
            cids = list(self._tiers[t])
            cids.sort(key=lambda cid: float(self._latency_sum_s.get(cid, 0.0)))
            if not cids:
                logging.info("[TiFL] Tier %s clients=[]", t)
                continue
            details = ", ".join(
                f"{cid}(avg={self._latency_sum_s.get(cid, 0.0)/max(1,int(self._latency_rounds.get(cid,0))):.3f}s "
                f"sum={self._latency_sum_s.get(cid,0.0):.3f}s r={int(self._latency_rounds.get(cid,0))})"
                for cid in cids
            )
            logging.info("[TiFL] Tier %s clients=%s", t, details)

    # ----------------------------- Scheduler (Algorithm 2) ----------------------------- #
    def _change_probs(self, accuracies: List[Optional[float]]) -> List[float]:
        """Implement ChangeProbs: lower-accuracy tiers get higher selection probability.

        The paper does not define a concrete formula; we use (1 - acc) as weights.
        """
        eps = 1e-6
        weights: List[float] = []
        for a in accuracies:
            if a is None:
                weights.append(1.0)
            else:
                weights.append(max(eps, 1.0 - float(a)))
        total = sum(weights)
        if total <= 0:
            return [1.0 / float(len(weights)) for _ in weights]
        return [w / total for w in weights]

    def _maybe_update_probs(self, server_round: int) -> None:
        """Update tier probabilities every I rounds, matching Algorithm 2 intent."""
        # We can only use accuracies from completed rounds, i.e., server_round-1.
        r = server_round - 1
        if r < self.prob_update_interval or (r % self.prob_update_interval) != 0:
            return
        cur = int(self._current_tier)
        prev = self._tier_accuracy_by_round.get(r)
        prev_i = self._tier_accuracy_by_round.get(r - self.prob_update_interval)
        if not prev or not prev_i:
            return
        a_cur = prev[cur] if cur < len(prev) else None
        a_cur_i = prev_i[cur] if cur < len(prev_i) else None
        if a_cur is None or a_cur_i is None:
            return
        if float(a_cur) <= float(a_cur_i):
            new_probs = self._change_probs(prev)
            self._tier_probs = list(new_probs)
            logging.info(
                "[TiFL] ChangeProbs at round=%s using A^r (r=%s): probs=%s",
                server_round,
                r,
                [f"{p:.3f}" for p in self._tier_probs],
            )

    def _sample_tier(self) -> int:
        """Sample a tier index according to current selection probabilities."""
        r = random.random()
        cum = 0.0
        for i, p in enumerate(self._tier_probs):
            cum += float(p)
            if r <= cum:
                return i
        return len(self._tier_probs) - 1

    def _select_clients_from_tier(
        self, available: Dict[str, ClientProxy], tier_idx: int, target: int
    ) -> List[ClientProxy]:
        cids = [cid for cid in self._tiers[tier_idx] if cid in available]
        random.shuffle(cids)
        selected: List[ClientProxy] = []
        for cid in cids[:target]:
            selected.append(available[cid])
        return selected

    # ----------------------------- Strategy hooks ----------------------------- #
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Provide a server-side timestamp so the client can estimate downlink latency.
        # This approximates "task issued by aggregator" in the TiFL response-latency definition.
        config = dict(config)
        config["server_send_time"] = float(time.time())

        parameters_to_send, payload_label, payload_bytes = self._prepare_parameters_to_send(parameters)
        payload_mb = payload_bytes / 1_000_000
        payload_mib = payload_bytes / (1024**2)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        if not client_manager.wait_for(min_num_clients, timeout=30):
            logging.warning(
                "[Server][TiFL] Round %s: only %s clients available after wait, need %s",
                server_round,
                client_manager.num_available(),
                min_num_clients,
            )
        raw_available = client_manager.all()
        available = {getattr(proxy, "cid", cid): proxy for cid, proxy in raw_available.items()}

        # Periodic reprofiling (optional)
        if (
            self.reprofiling_interval_rounds > 0
            and server_round > self.sync_rounds
            and (server_round % self.reprofiling_interval_rounds) == 0
        ):
            logging.info("[TiFL] Re-profiling triggered at round=%s", server_round)
            self._latency_sum_s = {}
            self._latency_rounds = {}
            self._dropouts = set()
            self._tier_of_cid = {}
            self._tiers = []
            self._tier_latency_avg_s = []

        # Profiling phase: assign profiling tasks (paper: ask clients, wait up to Tmax).
        if self._is_profiling_round(server_round):
            selected_clients = list(available.values())
            config = dict(config)
            config["tifl_phase"] = "profile"
            config["tifl_tmax_s"] = float(self.tmax_s)
            config["tifl_profile_warmup_rounds"] = int(self.profile_warmup_rounds)
        else:
            # Ensure tiers are built once after profiling completes.
            if not self._tiers:
                self._build_tiers(available)

            # Update tier selection probabilities every I rounds (Algorithm 2 intent).
            self._maybe_update_probs(server_round)

            # Select one tier (respecting credits) then sample |C| clients from it.
            selected_clients = []
            if self._tiers and self._tier_probs and self._tier_credits:
                # Paper assumes each tier has enough clients for |C|. In practice tiers can be sparse,
                # so we only consider tiers with enough clients to satisfy sample_size.
                eligible = [
                    t
                    for t in range(len(self._tiers))
                    if self._tier_credits[t] > 0 and len([cid for cid in self._tiers[t] if cid in available]) >= sample_size
                ]
                if not eligible:
                    eligible = [
                        t
                        for t in range(len(self._tiers))
                        if self._tier_credits[t] > 0 and len([cid for cid in self._tiers[t] if cid in available]) > 0
                    ]

                if eligible:
                    weights = [float(self._tier_probs[t]) for t in eligible]
                    total = sum(weights)
                    if total <= 0:
                        weights = [1.0 for _ in eligible]
                    tier_idx = random.choices(eligible, weights=weights, k=1)[0]
                    self._tier_credits[tier_idx] -= 1
                    self._current_tier = tier_idx
                    selected_clients = self._select_clients_from_tier(available, tier_idx, sample_size)

            # Fallback: if we couldn't get enough clients from a single tier, fall back to random sampling.
            if len(selected_clients) < sample_size:
                logging.warning(
                    "[TiFL] Round %s: tier=%s had only %s/%s clients available, falling back to random sampling",
                    server_round,
                    self._current_tier,
                    len(selected_clients),
                    sample_size,
                )
                self._current_tier = -1
                selected_clients = client_manager.sample(
                    num_clients=sample_size,
                    min_num_clients=min_num_clients,
                )

        if selected_clients:
            logging.info(
                "[Server][TiFL] Round %s: phase=%s sending %s payload %.3f MB (%.3f MiB) to %s clients "
                "(tiers=%s selected_tier=%s probs=%s credits=%s)",
                server_round,
                "profile" if self._is_profiling_round(server_round) else "train",
                payload_label,
                payload_mb,
                payload_mib,
                len(selected_clients),
                len(self._tiers) if self._tiers else 0,
                self._current_tier if self._tiers else -1,
                [f"{p:.3f}" for p in (self._tier_probs or [])],
                self._tier_credits if self._tier_credits else [],
            )

        fit_ins = FitIns(parameters_to_send, config)
        return [(client, fit_ins) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # During profiling, update L_i per the paper rule using client-observed fit elapsed time
        # (capped by Tmax) and treat failures as timeouts.
        #
        # IMPORTANT: TiFL "response latency" includes communication + compute. Flower's client-side
        # `fit()` does not include network I/O, so we compose:
        #   L_i ≈ downlink_s + local_fit_s + uplink_s
        # where:
        #   downlink_s is reported by the client as server_to_client_ms (based on server_send_time),
        #   local_fit_s is fit_elapsed_s (or train_time+edcode fallback),
        #   uplink_s is approximated on the server as (now - client_fit_end_time).
        if self._is_profiling_round(server_round):
            for client_proxy, fit_res in results:
                cid = getattr(client_proxy, "cid", None) or str(id(client_proxy))
                m = fit_res.metrics or {}
                observed_s = None
                # Use the same "upload time" approximation as your CSV pipeline:
                #   client_to_server_ms ≈ (server_arrival_time - client_fit_end_time) * 1000
                # Note: this mixes server/client clocks, but keeps behavior consistent with existing logs.
                local_fit_s = _safe_float(m.get("fit_elapsed_s"))
                if local_fit_s is None:
                    train_s = _safe_float(m.get("train_time")) or 0.0
                    edcode_s = _safe_float(m.get("edcode")) or 0.0
                    local_fit_s = float(train_s + edcode_s) if (train_s + edcode_s) > 0 else None

                down_ms = _safe_float(m.get("server_to_client_ms"))
                down_s = float(down_ms) / 1000.0 if down_ms is not None else 0.0

                upload_ms = _safe_float(m.get("client_to_server_ms"))
                if upload_ms is None:
                    server_arrival_time = _safe_float(m.get("server_arrival_time"))
                    client_fit_end_time = _safe_float(m.get("client_fit_end_time"))
                    if server_arrival_time is not None and client_fit_end_time is not None:
                        upload_ms = max(
                            0.0, (float(server_arrival_time) - float(client_fit_end_time)) * 1000.0
                        )
                up_s = float(upload_ms) / 1000.0 if upload_ms is not None else 0.0

                if local_fit_s is not None:
                    observed_s = float(down_s + float(local_fit_s) + float(up_s))

                logging.info(
                    "[TiFL][Profile] round=%s cid=%s down_s=%.3f local_s=%s up_s=%.3f observed_s=%s (tmax=%.3f)",
                    server_round,
                    cid,
                    float(down_s),
                    f"{float(local_fit_s):.3f}" if local_fit_s is not None else "NA",
                    float(up_s),
                    f"{float(observed_s):.3f}" if observed_s is not None else "NA",
                    float(self.tmax_s),
                )

                self._observe_latency_s(cid, observed_s, server_round=server_round)
            for failure in failures:
                if isinstance(failure, BaseException):
                    continue
                client_proxy, _ = failure
                cid = getattr(client_proxy, "cid", None) or str(id(client_proxy))
                logging.info(
                    "[TiFL][Profile] round=%s cid=%s observed_s=NA (failure/timeout, tmax=%.3f)",
                    server_round,
                    cid,
                    float(self.tmax_s),
                )
                self._observe_latency_s(cid, None, server_round=server_round)
        return super().aggregate_fit(server_round, results, failures)

    # ----------------------------- Per-tier evaluation ----------------------------- #
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Skip federated evaluate during profiling or before tiering exists.
        if self._is_profiling_round(server_round) or not self._tiers:
            return []

        config: Dict[str, Scalar] = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        # Paper evaluates on every client for every tier on their respective TestData_t.
        raw_available = client_manager.all()
        clients = list(raw_available.values())
        ins = EvaluateIns(parameters=parameters, config=config)
        return [(c, ins) for c in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        # Compute per-tier mean accuracy (unweighted mean across clients in tier).
        tier_acc_values: List[List[float]] = [[] for _ in range(self.num_tiers)]
        for client_proxy, eval_res in results:
            cid = getattr(client_proxy, "cid", None) or str(id(client_proxy))
            tier = self._tier_of_cid.get(cid)
            if tier is None:
                continue
            acc = _safe_float((eval_res.metrics or {}).get("accuracy"))
            if acc is None:
                continue
            tier_acc_values[tier].append(float(acc))

        tier_acc: List[Optional[float]] = []
        for t in range(self.num_tiers):
            vals = tier_acc_values[t]
            tier_acc.append(sum(vals) / float(len(vals)) if vals else None)
        self._tier_accuracy_by_round[int(server_round)] = tier_acc

        loss_agg, metrics_agg = super().aggregate_evaluate(server_round, results, failures)
        metrics_agg = dict(metrics_agg or {})
        for t, a in enumerate(tier_acc):
            if a is not None:
                metrics_agg[f"tifl_tier{t}_acc"] = float(a)
        metrics_agg["tifl_selected_tier"] = float(self._current_tier)
        return loss_agg, metrics_agg


def _safe_float(value: object) -> Optional[float]:
    """Convert to float safely."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_sample_counts(data_dir: Path) -> List[int]:
    """Load per-client sample counts from client_*.npz files (x_train length).

    This is only used to emulate FedCS paper's 'data size' resource information
    without relying on measured timing metrics.
    """
    if not data_dir.exists():
        return []
    files = sorted(data_dir.glob("client_*.npz"))
    counts: List[int] = []
    for fp in files:
        try:
            with np.load(fp) as npz:
                x = npz.get("x_train")
                if x is None:
                    continue
                counts.append(int(x.shape[0]))
        except Exception:
            continue
    # Shuffle to assign randomly to newly observed clients.
    rng = np.random.default_rng(1234)
    rng.shuffle(counts)
    return counts
