"""CPU calibration tests. Synthetic data only; no TensorFlow training or sudo."""

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.stats import logistic

from warmup.config import prepare_config
from warmup.controller import Calibration, read_timings
from warmup.export import export_last
from warmup.launch import client_command, launch_job, service_command
from warmup.measurement import MeasurementSession, atomic_json, load_cpu_map
from warmup.model import CpuModel, adjust_cpu, fit_cpu_model, fit_logistic, quantize_cpu
from warmup.simulation import simulated_stage


class ModelTests(unittest.TestCase):
    def test_logistic_recovery(self):
        values = logistic.ppf((np.arange(1000) + 0.5) / 1000, loc=10, scale=0.2)
        fit = fit_logistic(values)
        self.assertAlmostEqual(fit.theta_s, 10, places=4)
        self.assertLess(abs(fit.k_s / 0.2 - 1), 0.01)
        self.assertEqual(fit.n_samples, 1000)
        self.assertLess(fit.ks_distance, 0.005)

    def test_truncation_near_zero(self):
        f0 = logistic.cdf(0, loc=1, scale=0.8)
        values = logistic.ppf(f0 + (1 - f0) * (np.arange(2000) + 0.5) / 2000, loc=1, scale=0.8)
        fit = fit_logistic(values)
        self.assertLess(abs(fit.theta_s - 1), 0.01)
        self.assertLess(abs(fit.k_s - 0.8), 0.01)

    def test_bad_samples(self):
        for values in ([1, 2], [1, 1, 1], [0, 1, 2], [1, float("nan"), 2], [-1, 2, 3]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fit_logistic(values)

    def test_cpu_model_and_inverse(self):
        model = fit_cpu_model([(c, 4 * c ** -0.8 + 0.3) for c in (0.3, 0.5, 0.7, 0.9)])
        self.assertLess(model.relative_rmse, 1e-6)
        self.assertAlmostEqual(model.theta(0.6), 4 * 0.6 ** -0.8 + 0.3, places=5)
        self.assertAlmostEqual(model.inverse(model.theta(0.6)), 0.6, places=7)
        self.assertTrue(math.isinf(model.inverse(0)))

    def test_quantization(self):
        self.assertEqual(quantize_cpu(0.55555, 0.05, 1, 0.001), 0.556)
        self.assertEqual(quantize_cpu(0.001, 0.05, 1, 0.001), 0.05)
        self.assertEqual(quantize_cpu(float("inf"), 0.05, 1, 0.001), 1)

    def test_adjustment_direction_and_bounds(self):
        model = CpuModel(4, 0.8, 0.3, 0)
        target = model.theta(0.5)
        self.assertGreater(adjust_cpu(model, 0.5, target * 1.1, target, [], 0.05, 1, 0.001), 0.5)
        self.assertLess(adjust_cpu(model, 0.5, target * 0.9, target, [], 0.05, 1, 0.001), 0.5)
        with self.assertRaises(ValueError):
            adjust_cpu(model, 1, target * 2, target, [], 0.05, 1, 0.001)
        with self.assertRaises(ValueError):
            adjust_cpu(model, 0.05, target * 0.5, target, [], 0.05, 1, 0.001)

    def test_measured_bracket(self):
        model = CpuModel(4, 1, 0, 0)
        result = adjust_cpu(model, 0.5, 20, 8, [(0.6, 7)], 0.05, 1, 0.001)
        self.assertEqual(result, 0.55)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def test_record_full_precision_and_no_reuse(self):
        path = self.root / "timing.jsonl"
        session = MeasurementSession(path, "s1", "0")
        session.record(1, 1.123456789)
        self.assertEqual(json.loads(path.read_text())["train_time_s"], 1.123456789)
        with self.assertRaises(ValueError):
            MeasurementSession(path, "s2", "0")

    @patch("warmup.measurement.cpu_snapshot")
    def test_enforces_actual_quota_and_affinity(self, snapshot):
        session = MeasurementSession(self.root / "timing.jsonl", "s1", "0", 0.3, "2")
        snapshot.return_value = {"cpu_actual": 1.0, "cpu_affinity": [2]}
        with self.assertRaises(ValueError):
            session.record(1, 2)
        snapshot.return_value = {"cpu_actual": 0.3, "cpu_affinity": [3]}
        with self.assertRaises(ValueError):
            session.record(1, 2)
        self.assertFalse(session.output.exists())
        snapshot.return_value = {"cpu_actual": 0.3, "cpu_affinity": [2]}
        session.record(1, 2)

    def test_barrier_checks_stage(self):
        session = MeasurementSession(self.root / "timing.jsonl", "s1", "0")
        start = self.root / "start.json"
        atomic_json(start, {"stage_id": "wrong"})
        with self.assertRaises(ValueError):
            session.wait_for_start(self.root / "ready.json", start, 1)
        atomic_json(start, {"stage_id": "s1"})
        session.wait_for_start(self.root / "ready.json", start, 1)
        self.assertEqual(json.loads((self.root / "ready.json").read_text())["client_id"], "0")

    def test_cpu_map_named_columns_and_validation(self):
        path = self.root / "cpus.csv"
        path.write_text("theta_s,cpu_affinity,client_id,cpu\n10,2,00003,0.3\n")
        self.assertEqual(load_cpu_map(path), {"3": (0.3, "2")})
        path.write_text("client_id,new_cpu\n3,0.5\n")
        self.assertEqual(load_cpu_map(path), {"3": (0.5, "-")})
        for content in ("client_id,cpu\n0,30\n", "client_id,cpu\n0,0.3\n0,0.4\n",
                        "client_id,cpu\n0,nan\n", "client_id,theta\n0,1\n"):
            path.write_text(content)
            with self.assertRaises(ValueError):
                load_cpu_map(path)

    def test_sample_identity_and_discard(self):
        client = {"client_id": "0", "cpu": 0.3, "cpu_affinity": "2"}
        job = {"clients": [client], "stage_id": "scan_30/attempt_001", "rounds": 50,
               "output_dir": str(self.root)}
        simulated_stage(job)
        path = self.root / "client_0.jsonl"
        samples = read_timings(path, job["stage_id"], client, 50, 1)
        self.assertEqual(len(samples), 49)
        self.assertLess(max(samples), 10)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for field, wrong in (("round", 2), ("client_id", "1"), ("stage_id", "old"),
                             ("cpu_actual", 1), ("cpu_affinity", [3]), ("train_time_s", -1)):
            changed = [dict(row) for row in rows]
            changed[0][field] = wrong
            path.write_text("\n".join(json.dumps(row) for row in changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_timings(path, job["stage_id"], client, 50, 1)
        path.write_text("\n".join(json.dumps(row) for row in rows[:-1]))
        with self.assertRaises(ValueError):
            read_timings(path, job["stage_id"], client, 50, 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.cfg = prepare_config({"dataset": "femnist", "model": "cnn", "data_dir": "unused",
                                   "output_dir": str(self.root / "run")}, self.root, simulate=True)

    def test_pipeline_fixed_target_selective_adjustment_and_export(self):
        jobs = []
        def runner(job):
            jobs.append(job)
            simulated_stage(job)
        rows = Calibration(self.cfg, runner).run()
        self.assertEqual([j["clients"][0]["cpu"] for j in jobs[:4]], [0.3, 0.5, 0.7, 0.9])
        self.assertTrue(all(j["rounds"] == 50 for j in jobs[:4]))
        validations = jobs[4:]
        self.assertGreaterEqual(len(validations), 2)
        self.assertTrue(all(j["rounds"] == 30 for j in validations))
        first, second = validations[:2]
        self.assertNotEqual(first["clients"][0]["cpu"], second["clients"][0]["cpu"])
        self.assertEqual(first["clients"][1:], second["clients"][1:])
        self.assertTrue(all(abs(row["relative_error"]) <= 0.03 for row in rows))
        output = Path(self.cfg["output_dir"])
        with (output / "profiles.csv").open() as handle:
            profiles = list(csv.DictReader(handle))
        measured_target = max(float(row["theta_s"]) for row in profiles if float(row["cpu"]) == 0.9)
        self.assertEqual(rows[0]["theta_target_s"], measured_target)
        self.assertTrue(all(int(row["n_samples"]) == 49 for row in profiles))
        with (output / "validation_history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        self.assertTrue(all(int(row["n_samples"]) == 29 for row in history))
        self.assertEqual({float(row["theta_target_s"]) for row in history}, {measured_target})
        self.assertTrue((output / "simulated_cpu_config.csv").exists())
        self.assertFalse((output / "final_cpu_config.csv").exists())
        self.assertFalse((output / "launch_final_clients.sh").exists())

    def test_resume_reuses_completed_batches(self):
        expected = Calibration(self.cfg, simulated_stage).run()
        with patch("warmup.controller.external_stage", side_effect=AssertionError("No launching")) as runner:
            actual = Calibration(self.cfg, runner).run(resume=True)
            runner.assert_not_called()
        self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            Calibration(self.cfg, simulated_stage).run()
        changed = {**self.cfg, "lr": 0.02}
        with self.assertRaises(ValueError):
            Calibration(changed, simulated_stage).run(resume=True)

    def test_partial_stage_retry_preserves_failed_attempt(self):
        def failing(job):
            if job["stage_id"].startswith("scan_50"):
                raise RuntimeError("Synthetic interruption")
            simulated_stage(job)
        with self.assertRaises(RuntimeError):
            Calibration(self.cfg, failing).run()
        stages = Path(self.cfg["output_dir"]) / "stages"
        self.assertTrue((stages / "scan_30/complete.json").exists())
        self.assertFalse((stages / "scan_50/complete.json").exists())
        Calibration(self.cfg, simulated_stage).run(resume=True)
        self.assertTrue((stages / "scan_50/attempt_001").exists())
        self.assertTrue((stages / "scan_50/attempt_002").exists())

    def test_iteration_limit_exports_last_measured_cpu_with_warning(self):
        self.cfg["max_iterations"] = 1
        self.cfg["simulation"] = False
        jobs = []
        def runner(job):
            jobs.append(job)
            simulated_stage(job)
        rows = Calibration(self.cfg, runner).run()
        output = Path(self.cfg["output_dir"])
        status = json.loads((output / "status.json").read_text())
        self.assertEqual(status["status"], "max_iterations_reached")
        self.assertFalse(status["converged"])
        self.assertEqual(status["failing_clients"], ["0"])
        self.assertEqual({row["client_id"]: row["cpu"] for row in rows},
                         {client["client_id"]: client["cpu"] for client in jobs[-1]["clients"]})
        self.assertFalse(rows[0]["passed"])
        self.assertTrue(all(row["passed"] for row in rows[1:]))
        self.assertTrue(all(not row["converged"] and row["validation_iteration"] == 1 for row in rows))
        self.assertTrue((output / "final_cpu_config.csv").exists())
        self.assertIn("WARNING", (output / "launch_final_clients.sh").read_text())
        self.assertFalse((output / "stages/validate_001/adjustments.json").exists())

    def test_training_failure_still_does_not_export_final(self):
        self.cfg["max_iterations"] = 1
        def failing(job):
            if job["stage_id"].startswith("validate_"):
                raise RuntimeError("Training process failed")
            simulated_stage(job)
        with self.assertRaises(RuntimeError):
            Calibration(self.cfg, failing).run()
        output = Path(self.cfg["output_dir"])
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
        self.assertFalse((output / "final_cpu_config.csv").exists())
        self.assertFalse((output / "simulated_cpu_config.csv").exists())

    def test_launcher_command_has_real_quota_and_cpu_only_workload(self):
        job = {**self.cfg, "rounds": 50, "stage_id": "scan_30"}
        for cpu in (0.3, 0.5, 0.7, 0.9):
            client = {"client_id": "1", "cpu": cpu, "cpu_affinity": "2"}
            command = service_command(job, client, self.root, "unique.service", ["sudo", "-n"])
            self.assertIn(f"CPUQuota={cpu * 100:.2f}%", command)
            self.assertIn("--setenv=CUDA_VISIBLE_DEVICES=", command)
            self.assertIn("--uid=" + __import__("getpass").getuser(), command)
            self.assertNotIn("CPUQuota=100%", command)
            self.assertIn("--wait", command)
            self.assertIn("--collect", command)
            client_cmd = client_command(job, client, self.root)
            self.assertEqual(client_cmd[:3], ["taskset", "-c", "2"])
            self.assertIn("--local-only", client_cmd)
            self.assertNotIn("--pacer", client_cmd)
            self.assertEqual(client_cmd[client_cmd.index("--epochs") + 1], "5")

    def test_config_rejects_bad_values(self):
        base = {"dataset": "femnist", "model": "cnn", "data_dir": "unused", "output_dir": "output"}
        for field, value in (("min_cpu", 0.01), ("cpu_ids", [1, 1, 2]), ("tolerance", 3),
                             ("lr", float("nan")), ("client_ids", [0, "00"]),
                             ("validation_rounds", 3), ("unknown", 1), ("cpu_step", 0.0001)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_config({**base, field: value}, self.root, simulate=True)

    def test_cli_dry_run_simulation_and_resume_never_launch_clients(self):
        project = Path(__file__).resolve().parents[1]
        data = self.root / "data"
        data.mkdir()
        np.savez(data / "client_00000.npz", x_train=np.zeros((2, 2)), y_train=np.zeros(2))
        config_path = self.root / "config.json"
        output = self.root / "cli_output"
        atomic_json(config_path, {"dataset": "mnist", "model": "cnn", "client_ids": [0],
                                  "data_dir": str(data), "output_dir": str(output)})
        command = [sys.executable, str(project / "warmup_control.py"), "--config", str(config_path)]
        env = {**os.environ, "PATH": "/nonexistent", "PYTHONDONTWRITEBYTECODE": "1"}
        dry = subprocess.run(command, env=env, check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(dry.stdout)["mode"], "dry-run")
        self.assertFalse(output.exists())
        subprocess.run([*command, "--simulate"], env=env, check=True, capture_output=True, timeout=30)
        self.assertFalse(output.exists())
        simulated = Path(str(output) + ".simulation")
        self.assertTrue((simulated / "simulated_cpu_config.csv").exists())
        self.assertFalse((simulated / "final_cpu_config.csv").exists())
        resumed = subprocess.run([*command, "--simulate", "--resume"], env=env, check=True,
                                 capture_output=True, text=True, timeout=30)
        self.assertIn("Reuse scan_30", resumed.stdout)

    def test_iteration_budget_can_be_extended_on_resume(self):
        self.cfg["max_iterations"] = 1
        self.assertFalse(Calibration(self.cfg, simulated_stage).run()[0]["converged"])
        self.cfg["max_iterations"] = 3
        Calibration(self.cfg, simulated_stage).run(resume=True)
        self.assertEqual(json.loads((Path(self.cfg["output_dir"]) / "status.json").read_text())["status"], "converged")

    def test_real_export_preserves_workload_and_rejects_corrupt_resume(self):
        # Only the output naming is real; the injected runner still performs no training.
        self.cfg["simulation"] = False
        Calibration(self.cfg, simulated_stage).run()
        output = Path(self.cfg["output_dir"])
        self.assertTrue((output / "final_cpu_config.csv").exists())
        launch = (output / "launch_final_clients.sh").read_text()
        for fragment in ("ENABLE_CPU_AFFINITY=0", "CPU_MAP_ONLY=1", "LOCAL_EPOCHS=5", "BATCH_SIZE=8", '"$1" 0'):
            self.assertIn(fragment, launch)
        self.assertNotIn("CPU_ONLY=1", launch)
        damaged = output / "stages/validate_002/attempt_001/client_0.jsonl"
        damaged.write_text("{}\n")
        with self.assertRaises(ValueError):
            Calibration(self.cfg, simulated_stage).run(resume=True)
        self.assertFalse((output / "final_cpu_config.csv").exists())
        self.assertFalse((output / "launch_final_clients.sh").exists())

    def test_export_old_run_without_retraining_refitting_or_changing_measurements(self):
        self.cfg["max_iterations"] = 1
        self.cfg["simulation"] = False
        def runner(job):
            atomic_json(Path(job["output_dir"]) / "job.json", job)
            simulated_stage(job)
        expected = Calibration(self.cfg, runner).run()
        output = Path(self.cfg["output_dir"])
        for name in ("final_cpu_config.csv", "launch_final_clients.sh"):
            (output / name).unlink()
        atomic_json(output / "status.json", {"status": "failed", "error": "No convergence after 1 validation batches"})
        originals = {path: path.read_bytes() for path in output.rglob("*.jsonl")}
        for path in (output / "target.json", output / "validation_history.csv",
                     output / "stages/validate_001/attempt_001/fits.json", output / "manifest.json"):
            originals[path] = path.read_bytes()
        with patch("warmup.controller.external_stage", side_effect=AssertionError("No training")), \
                patch("warmup.controller.fit_logistic", side_effect=AssertionError("No refitting")):
            rows = export_last(output)
        self.assertEqual(rows, expected)
        self.assertTrue(all(path.read_bytes() == content for path, content in originals.items()))
        self.assertEqual(json.loads((output / "status.before_export.json").read_text())["status"], "failed")
        self.assertFalse(json.loads((output / "last_export.json").read_text())["retrained"])
        self.assertTrue((output / "launch_final_clients.sh").exists())

    def test_export_last_rejects_incomplete_validation(self):
        def failing(job):
            simulated_stage(job)
            if job["stage_id"].startswith("validate_"):
                raise RuntimeError("Interrupted before all clients finished")
        with self.assertRaises(RuntimeError):
            Calibration(self.cfg, failing).run()
        output = Path(self.cfg["output_dir"])
        with self.assertRaises(ValueError):
            export_last(output)
        self.assertFalse((output / "simulated_cpu_config.csv").exists())

    def test_export_last_rejects_mutated_history(self):
        self.cfg["max_iterations"] = 1
        Calibration(self.cfg, simulated_stage).run()
        output = Path(self.cfg["output_dir"])
        (output / "simulated_cpu_config.csv").unlink()
        history_path = output / "validation_history.csv"
        with history_path.open() as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["theta_s"] = str(float(rows[0]["theta_s"]) + 1)
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            export_last(output)
        self.assertFalse((output / "simulated_cpu_config.csv").exists())

    def test_cli_limit_and_export_last_report_nonconvergence(self):
        project = Path(__file__).resolve().parents[1]
        config_path = self.root / "cli_limit.json"
        output = self.root / "cli_limit"
        atomic_json(config_path, {"dataset": "femnist", "model": "cnn", "data_dir": "unused",
                                  "output_dir": str(output), "client_ids": [0, 1, 2], "max_iterations": 1})
        command = [sys.executable, str(project / "warmup_control.py")]
        env = {**os.environ, "PATH": "/nonexistent", "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([*command, "--config", str(config_path), "--simulate"],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        simulated = Path(str(output) + ".simulation")
        (simulated / "simulated_cpu_config.csv").unlink()
        exported = subprocess.run([*command, "--export-last", str(simulated)],
                                  env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(exported.returncode, 2, exported.stdout + exported.stderr)
        self.assertTrue((simulated / "simulated_cpu_config.csv").exists())
        self.assertFalse((simulated / "final_cpu_config.csv").exists())

    @patch("warmup.launch.shutil.which", return_value="/usr/bin/mock")
    @patch("warmup.launch.subprocess.run")
    @patch("warmup.launch.subprocess.Popen")
    def test_launcher_failure_stops_only_owned_units(self, popen, run, which):
        process = MagicMock()
        process.poll.return_value = 1
        popen.return_value = process
        job = {**self.cfg, "rounds": 50, "stage_id": "scan_30",
               "clients": [{"client_id": "0", "cpu": 0.3, "cpu_affinity": "2"}]}
        with self.assertRaises(RuntimeError):
            launch_job(job)
        unit = json.loads((Path(job["output_dir"]) / "units.json").read_text())["units"][0]
        self.assertTrue(unit.startswith("flower-warmup-"))
        stop = [call.args[0] for call in run.call_args_list if "systemctl" in call.args[0]]
        self.assertEqual(len(stop), 1)
        self.assertEqual(stop[0][-2:], ["stop", unit])
        process.wait.assert_called()


if __name__ == "__main__":
    unittest.main()
