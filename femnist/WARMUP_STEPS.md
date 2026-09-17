# Fixed-step warm-up with speed heterogeneity

This is an opt-in extension on branch `step-warmup`. The existing epoch config
`examples/warmup.femnist20.json` and its 90%-CPU target initialization are unchanged.
The current executor starts a real Flower server for each stage, then calls the
existing `launch_clients.sh`. It does not use the older local-only launch path.

## Configuration

New example: `examples/warmup.steps.femnist20.json`.

- `training_mode: "steps"`, `local_steps: 20`: exactly 20 optimizer updates per
  client per round. Keras uses `epochs=1, steps_per_epoch=20`; the existing
  `epochs: 10` and `server_local_epochs: 20` fields are retained but inactive.
- Batch size 64, LR 0.003, FEMNIST/CNN, client IDs 0-19, ports, MPS setting,
  server affinity `20-31,52-63`, and server CSV `logs/atest.csv` match the old example.
- CPU scan: 30%, 50%, 70%, 90%, each 50 rounds; discard round 1 from fitting,
  but keep it in the raw log and mark it `used_for_fit=False` in the exported CSV.
- `heterogeneity.reference_client_id: 0`: the common baseline is this client's
  **measured** speed at 50% CPU, not the mean across clients or a fitted prediction.
- `distribution: "normal"`, `mean: 1.0`, `variance: 0.04`, `seed: 42`: sample
  dimensionless speed factors once. Requested speed = baseline speed * factor.
  The normal standard deviation is sqrt(variance), here 0.2.
- `homogeneous` requires variance 0; all clients request the same speed, but each
  uses its own CPU-speed model. `exponential` requires variance = mean squared
  (or omit variance to derive it). Finite samples need not have exactly the
  configured population mean and variance.
- Nonpositive or unreachable requested speeds stop initialization and leave a
  feasibility report in `heterogeneity_plan.json`. No clipping or resampling.
  Change the distribution settings and use a new output directory to retry.
- `heterogeneity.initial_rounds: 30`, `initial_discard: 1`: measure the initial
  heterogeneous allocation. The largest fitted Logistic theta from this stage
  is the anchor. With `q=0.9`, `Gamma=client count`, and `ref_a=0.02`, compute
  the round deadline, round it upward to the next 0.5 seconds, then invert that
  rounded deadline to obtain the fixed validation theta target.
- Validation: 30 rounds, discard first, tolerance 0.03, at most 5 batches. Only
  clients exceeding the tolerance change CPU for the following batch.

`enable_cpu_affinity` stays **false**, matching the user's configuration. A 50%
CPU quota is a time budget equivalent to half of one CPU, not a dedicated core,
nor a guarantee of measured 50% utilization. Fixed K and batch size do not imply
identical measured steps/s. Set affinity explicitly for a new experiment if needed;
do not reuse profiles measured with another resource configuration.

## Run

On 4a6, check the config without starting any processes:

```bash
cd /home/xiaoyan/wholeflower/femnist
../venv/bin/python warmup_control.py --config examples/warmup.steps.femnist20.json
```

Check the new example's K, distribution parameters and output directory before
starting. Stop any experiment using the same server port or `fl_client_<id>.scope`
names. Keep the existing sudo/systemd setup available to `launch_clients.sh`.
Do not launch another server or the clients separately during warm-up.

```bash
../venv/bin/python warmup_control.py \
  --config examples/warmup.steps.femnist20.json --execute
```

Resume an interrupted run with unchanged workload, code, dataset metadata and
distribution settings:

```bash
../venv/bin/python warmup_control.py \
  --config examples/warmup.steps.femnist20.json --execute --resume
```

Completed batches are checked and reused; an incomplete batch gets a new attempt
directory. The saved distribution is not sampled again, and target is checked
against its source measurement. Old epoch runs cannot be resumed as step runs.

For a synthetic control-flow test, use `--simulate` instead of `--execute`.
It writes to `<output_dir>.simulation`, never starts clients, and only creates
`simulated_cpu_config.csv`, not a real final configuration or launcher.

## Output

Default real output directory:
`/home/xiaoyan/wholeflower/femnist/logs/mnistdata/warmup_femnist20_steps_run01`.

| File | Meaning |
| --- | --- |
| `speed_profiles.csv` | Measured speed for each client and CPU scan level: total actual steps / total retained training time. Also includes round-speed mean, median and variance. |
| `speed_cpu_models.json` | Per-client model: seconds/step = a * CPU^(-beta) + floor, and relative fitting error. |
| `profiles.csv`, `cpu_models.json` | Separate Logistic theta fits and CPU-theta models, used by the existing adjustment loop. |
| `heterogeneity_plan.json` | Reference speed, factors, seed, requested speeds, feasible bounds, quantization details and checksum. |
| `heterogeneous_cpu_config.csv` | Initial heterogeneous CPU allocation; never replaced by the equalized final allocation. |
| `heterogeneous_initial_measurements.csv` | Initial measured speed, requested-speed error, theta and Logistic scale. |
| `target.json` | Slowest initial theta anchor, raw/rounded deadline, inverted fixed theta target and source-stage fingerprint. |
| `pacer-control.json` | Pacer schema v1 configuration generated from the same theta/deadline pair after a real warm-up finishes. |
| `initial_cpu_config.csv` | First equalization allocation, used by validate_001. |
| `validation_history.csv` | Per-validation theta, CPU, fixed target and relative error. |
| `final_cpu_config.csv`, `launch_final_clients.sh` | Last measured allocation and a launcher carrying the same K, batch size and step seed. |
| `status.json` | Converged flag, failing clients, final validation iteration and export reason. |
| `stages/<stage>/attempt_*/` | Original client JSONL, process/training logs, server.log, launcher.log, job.json, CPU map, fits.json and speeds.json. Failed attempts remain. |
| `timing_exports/*.csv` | All recorded rounds, including discarded rounds, with step/speed fields. |

Stage names: `scan_30`, `scan_50`, `scan_70`, `scan_90`,
`heterogeneous_initial`, and `validate_001` through `validate_005`.
The configured shared server CSV is unchanged in location and schema. Use the
per-stage JSONL/exports for fitting; do not mix shared server rows by round alone.

Reaching the iteration limit still exports the **last measured** config but sets
`converged=false` and exits 2. This does not mean every client meets 3%.
Success exits 0; training failure, incomplete measurements or invalid config exits
1 and does not produce a new final config. A target outside adjustable CPU bounds
is also a failure; target is never changed to hide it.

To re-export an already completed validation without training:

```bash
../venv/bin/python warmup_control.py --export-last \
  logs/mnistdata/warmup_femnist20_steps_run01
```

## Formal Experiment

After warm-up, inspect `status.json`. Start the server in one terminal using the
same K, batch size, learning rate, dataset, model and server affinity:

```bash
cd /home/xiaoyan/wholeflower/femnist
taskset -c 20-31,52-63 env CUDA_VISIBLE_DEVICES= \
  /home/xiaoyan/wholeflower/venv/bin/python -m run_server \
  --dataset femnist --model cnn --clients 20 --rounds 50 \
  --reporting-fraction 1.0 --address 0.0.0.0:8081 \
  --client-lr 0.003 --batch-size 64 --local-epochs 20 \
  --local-steps 20 --step-seed 42 --downlink-num-bits 0 \
  --csv-path logs/mnistdata/step_calibrated.csv
```

In a second terminal, start the generated client launcher:

```bash
cd /home/xiaoyan/wholeflower/femnist
bash logs/mnistdata/warmup_femnist20_steps_run01/launch_final_clients.sh \
  127.0.0.1:8081
```

These commands are for the new example, not arbitrary previous configs. Change
both ends consistently when changing K or step seed; mismatches fail explicitly.
CPU/GPU environment overrides such as `CPU_ONLY` should also be identical between
warm-up and the formal experiment. This extension does not change their defaults.

## Measurement Contract

Fixed steps currently support the array-backed FEMNIST/MNIST/FMNIST/CIFAR10 and
StackOverflow input paths, not streaming speech or Shakespeare. The data stream
uses shuffle -> repeat -> full batches. Sampling depends on step seed, client ID
and server round. It does not change the existing model initialization policy.
`model.optimizer.iterations` must increase by K, otherwise the measurement fails.
The timing definition is `model_fit_fixed_steps_v1`: dataset construction is
outside the timer; batch fetching and training inside model.fit are included.
Flower aggregation weights remain local partition sizes, not K * batch size.

Speed means actual steps / elapsed training time, not K / Logistic theta.
CPU-speed curves initialize the heterogeneous allocation; the separate CPU-theta
curves drive equalization. Equalizing theta with fixed K generally reduces the
initial speed heterogeneity. The final config is not claimed to retain the
sampled normal/exponential distribution.

## Tests

```bash
cd /home/xiaoyan/wholeflower/femnist
env CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 \
  TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  FLWR_TELEMETRY_ENABLED=0 PYTHONDONTWRITEBYTECODE=1 \
  ../venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q
```

Tests include tiny real TensorFlow models and synthetic controller stages, not a
full federated experiment. Passing these tests does not establish real CPU-speed
curves or guarantee 3% convergence under the machine's current load.
