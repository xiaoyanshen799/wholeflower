# FedPacer runtime parameter layer

This integration targets the `femnist/run_server.py` and `femnist/run_client.py`
entry points with the project's **Flower 1.5.0** `venv`. It supports FedAvg,
FedAvgM and the existing QuantizedFedAvgM. The other entry points and
`.venv_ixi` are not wired to this integration.

The external program supplies `theta_target_s`, `deadline_s`, `ref_a`, and
probability bounds. The client estimates a single gamma from the timings of
normal training rounds. The server sums valid gammas, evaluates the reference
CDF at the deadline, and emits a recalibration request after persistent drift.
It never computes or changes the target itself, performs warm-up training, or
changes CPU/GPU allocations. Model parameters and compression follow the
existing training path.

## Running

The values in `examples/pacer-control.example.json` are **illustrative**, not
profiled settings for the installed datasets. For an actual experiment point
`--pacer-config` at the JSON produced by your external parameter program.

From `/home/xiaoyan/wholeflower/femnist`, a server invocation is:

```bash
../venv/bin/python run_server.py \
  --dataset cifar10 --model cnn --clients 3 --rounds 100 \
  --reporting-fraction 1.0 --strategy fedavg \
  --pacer-config examples/pacer-control.example.json \
  --pacer-log logs/pacer_rounds.jsonl
```

Start each client with its own stable ID and data partition. For example, on
a client machine with the project and dependencies installed:

```bash
../venv/bin/python run_client.py \
  --cid 0 --server SERVER_HOST:8081 \
  --dataset cifar10 --model cnn --data-dir data_partitions_cifar10 \
  --uplink-num-bits 0 --pacer
```

Use IDs `0`, `1`, `2` for the sample roster. Each process writes its own
`logs/pacer_client_<cid>.jsonl`; `--pacer-log` overrides the path. Client logs
also expose the received target as a `target_received` record. No extra
network endpoint is used; all control fields travel with Flower's normal fit
messages. The existing `--local-only` profiling mode is separate from `--pacer`.

All required clients must participate each round. The server rejects TiFL,
FedCS, sampling fractions below 1, and inconsistent roster sizes in Pacer mode.
Incomplete training results abort the run instead of partially aggregating.
Missing/unusable gamma does not discard an otherwise valid model update.

## Configuration lifecycle

- Set a distinct `run_id` for each new experiment, and start at `effective_round=1`.
- `config_version` identifies immutable settings. To update target or other
  settings, publish a higher version and the earliest `effective_round`.
- Write a complete temporary JSON file, then atomically rename it over the
  configured file. The server reads once at each round boundary. All clients
  receive the same version for that round.
- Same-version changes, backwards versions and malformed files are rejected;
  the last accepted configuration remains active. An already accepted future
  version remains pending even if a later file read fails.
- Changing the run ID or required client roster requires a new run.
- Every version change resets the client window and server monitoring streak.
  Targets are never computed by the runtime.

`q_lower` and `q_upper` are a **probability tolerance band**, not a statistical
confidence interval. The example `[0.85, 0.95]` does not guarantee `p >= 0.90`.
Use `[q, q_upper]` when the lower acceptance bound should equal `q`. These are
model estimates under the paper's approximation/independence assumptions, not
statistical certification of an end-to-end latency SLO.

## Online gamma

The client retains up to `gamma_window` ordinary training durations. Once it
has `gamma_min_samples`, it fits only the positive exponent in
`F_star(t; theta_target, ref_a) ** gamma` to the empirical CDF, using SciPy
`curve_fit`. The reference location and scale are fixed, with
`k_star = ref_a * theta_target_s`. ECDF ties are handled explicitly; no slow
samples are discarded. Bounds `[1e-6, 1e6]` are numerical guards; a fit at a
guard, an unidentifiable fit, or an invalid/constant sample window is reported
as unusable feedback.

The default window (50), minimum (20), and patience (3) are configurable
engineering choices, not values mandated by the paper. Until the window is
ready, the client returns `pacer.feedback_valid=false`, not a fabricated
`gamma=1`. Server state is `insufficient_feedback` until the whole roster is
ready. Fit RMSE is included for diagnosis; it is not a confidence interval or
a proof that the model is suitable for a particular workload.

The window also resets when observed epochs, batch size, sample count or an
acknowledged resource revision changes. An externally changed workload or
resource allocation that is not otherwise detectable must be accompanied by
a configuration/resource revision change.

## External resource acknowledgement (optional)

Without `--pacer-resource-state`, the client marks resources `external_managed`:
it reports targets and assumes the operator manages allocation separately. It
does **not** claim to have applied a quota. To verify acknowledgement from an
external allocator, supply a local JSON file with:

```json
{
  "run_id": "pacer-example",
  "config_version": 1,
  "client_id": "0",
  "resource_version": 1,
  "status": "applied"
}
```

The allocator publishes this file atomically after applying resources. The
client reads it before and after training. Missing, malformed, mismatched or
changed state makes that round's gamma invalid and clears the estimator;
ordinary training still proceeds. Increase `resource_version` whenever
allocation changes. `target_received` is emitted before this check, allowing
an external program to observe new targets without interpreting model data.
This file is an acknowledgement contract, not a kernel quota readback.

## Logs and decisions

`logs/pacer_rounds.jsonl` is independent of the existing CSV files. Each
`round_observation` identifies run, round and config version, valid feedback
count, gamma sum, estimated `p_hat`, and state:

- `insufficient_feedback`: one or more clients lack usable feedback.
- `within_band`: `q_lower <= p_hat <= q_upper`.
- `below_band` / `above_band`: estimate outside the tolerance band.
- `incomplete_round`: missing/failed required training results or invalid identity.

Consecutive fresh complete feedback sets on the same side count toward
`violation_patience`. Cached evidence cannot increment the streak. A trigger
produces an additional `recalibration_requested` record; it does not stop
training or write a new target. The external controller consumes the event
and may publish a higher configuration version.

`local_train_round_s = max(client train_time)` and `observed_deadline_met` are
logged separately from model-derived `p_hat`. They cover the configured
**local training wall-time** measurement, including the dataset/validation
work already inside this project's timing segment. They exclude network and
server aggregation time. Offline reference fitting must use that same scope.

## Installed Flower patch

The project's `venv/lib/python3.10/site-packages/flwr/server/server.py` retains
the pre-existing `[SEND]`, `[SUBMIT]` and `[DISPATCH]` logging. The Pacer patch:

- Copies each client's mutable FitIns/config before thread submission.
- Restores `Code.OK` checking and Flower-compatible failure results.
- Measures queue wait and RPC duration with a monotonic clock.
- Records `server_arrival_time` immediately when the worker RPC returns,
  separately from `server_collection_time` in the collecting thread.
- Adds `pacer.client_rpc_elapsed_s` and, on complete rounds,
  `pacer.dispatch_barrier_elapsed_s` to the server-side result metrics.

`server_complete_mono` is a server-local timing coordinate; do not compare it
to a clock on another host. The previous cross-host timestamp differences
remain approximate diagnostic metrics.

The pre-change file, including your existing logging, is preserved in
`../patches/flower-server.pre-pacer.py.bak`. The incremental patch is in
`../patches/flower-1.5.0-pacer-timing.patch`; it applies to that customized
baseline, **not** to an arbitrary pristine Flower installation. Preview with:

```bash
patch --dry-run -p1 -d ../venv/lib/python3.10/site-packages \
  < ../patches/flower-1.5.0-pacer-timing.patch
```

It is already applied on 4a6. Do not reinstall Flower over your logging patch.
Remote clients need the updated `femnist/pacer` package and entry point;
only the server needs this server-side venv patch.

## Verification

From `/home/xiaoyan/wholeflower/femnist`:

```bash
FLWR_TELEMETRY_ENABLED=0 TF_CPP_MIN_LOG_LEVEL=3 \
  ../venv/bin/python -m unittest discover -s tests -p 'test_pacer*.py' -v

../venv/bin/python tests/pacer_grpc_smoke.py
```

Unit tests cover actual compressed/uncompressed client branches, scalar
serialization, gamma recovery, invalid/old feedback, config isolation,
monitoring and weighted aggregation. The network smoke test runs three real
gRPC clients for eight rounds and externally updates the target for round 5.
It uses synthetic weights/timings, cleans up its processes and temporary logs,
and does not train a model or change resource allocations.
