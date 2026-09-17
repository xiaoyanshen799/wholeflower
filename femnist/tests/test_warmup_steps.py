"""Fixed-step speed calibration; synthetic stages, no services or full experiment."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from fixed_step_training import TIMING_DEFINITION, resolve_seed, resolve_steps, with_fixed_steps
from warmup.config import prepare_config
from warmup.controller import Calibration, batch_key, external_stage
from warmup.export import export_last
from warmup.heterogeneity import (build_speed_plan, prepare_heterogeneity, read_plan,
                                  sample_speed_factors, validate_speed_target)
from warmup.measurement import atomic_json
from warmup.pacer_target import ceil_to_quantum, derive_pacer_target
from warmup.simulation import simulated_stage
from warmup.speed import SpeedCpuModel, fit_speed_cpu_model, summarize_speed, validate_step_record
from pacer.protocol import ControlConfig


def timing(duration, steps=20, batch=64):
    return {"train_time_s": duration, "timing_definition": TIMING_DEFINITION,
            "metrics": {"training_mode": "steps", "timing_definition": TIMING_DEFINITION,
                        "local_steps_requested": steps, "local_steps_used": steps,
                        "batch_size_used": batch, "processed_examples": steps * batch,
                        "steps_per_second": steps / duration, "step_seed": 42}}


class WorkloadTests(unittest.TestCase):
    def test_rounded_deadline_and_inverse_theta(self):
        self.assertEqual(ceil_to_quantum(6.2), 6.5)
        self.assertEqual(ceil_to_quantum(6.7), 7.0)
        self.assertEqual(ceil_to_quantum(7.0), 7.0)
        result = derive_pacer_target(4.970102012501917, 20)
        self.assertEqual(result["deadline_s"], 5.5)
        log_q_per_client = math.log(0.9) / 20
        logit = log_q_per_client - math.log(-math.expm1(log_q_per_client))
        self.assertAlmostEqual(result["theta_target_s"] * (1 + 0.02 * logit), 5.5)
        self.assertGreaterEqual(result["theta_target_s"], result["anchor_theta_s"])

    def test_steps_and_seed_resolution(self):
        self.assertIsNone(resolve_steps(None, None))
        self.assertEqual(resolve_steps(None, 20), 20)
        self.assertEqual(resolve_steps(20, None), 20)
        self.assertEqual(resolve_steps(20, 20), 20)
        for invalid in (0, -1, True, 1.5, "20"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                resolve_steps(invalid, None)
        with self.assertRaises(ValueError):
            resolve_steps(20, 30)
        self.assertEqual(resolve_seed(None, 123), 123)
        self.assertEqual(resolve_seed(None), 42)
        for invalid in (-1, 2**32, True, 1.1):
            with self.assertRaises(ValueError):
                resolve_seed(invalid)
        with self.assertRaises(ValueError):
            resolve_seed(42, 43)

    def test_server_config_preserves_base_values(self):
        original = {"local_epochs": 10, "batch_size": 64, "custom_key": "preserved"}
        fn = with_fixed_steps(lambda r: original, 20, 42)
        actual = fn(7)
        self.assertEqual(actual, {**original, "local_steps": 20, "training_mode": "steps",
                                  "step_seed": 42, "server_round": 7})
        self.assertNotIn("local_steps", original)

    def test_speed_is_total_steps_over_total_time(self):
        result = summarize_speed([timing(1), timing(2), timing(3)])
        self.assertEqual(result["speed_steps_per_s"], 10)
        self.assertNotEqual(result["speed_steps_per_s"], result["round_speed_mean"])
        self.assertEqual(result["total_steps"], 60)
        self.assertEqual(result["total_train_time_s"], 6)

    def test_step_metadata_is_verified(self):
        validate_step_record(timing(2), 20, 64, 42)
        for key, value in (("local_steps_used", 19), ("local_steps_requested", 19),
                           ("batch_size_used", 8), ("processed_examples", 20),
                           ("steps_per_second", 123), ("step_seed", 43),
                           ("training_mode", "epochs"), ("timing_definition", "model_fit_v2")):
            row = timing(2)
            row["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_step_record(row, 20, 64, 42)
        row = timing(2)
        row["timing_definition"] = "model_fit_v2"
        with self.assertRaises(ValueError):
            validate_step_record(row, 20)

    def test_speed_model_inverse(self):
        expected = SpeedCpuModel(0.2, 0.8, 0.015, 0)
        actual = fit_speed_cpu_model([(c, expected.speed(c)) for c in (0.3, 0.5, 0.7, 0.9)])
        self.assertLess(actual.relative_rmse, 1e-6)
        self.assertAlmostEqual(actual.speed(0.6), expected.speed(0.6), places=5)
        self.assertAlmostEqual(actual.inverse(actual.speed(0.6)), 0.6, places=6)
        for invalid in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                actual.inverse(invalid)


class ConfigAndPlanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw = {"dataset": "femnist", "model": "cnn", "data_dir": "unused", "output_dir": "run",
                    "training_mode": "steps", "local_steps": 20, "batch_size": 64,
                    "heterogeneity": {"distribution": "homogeneous"}}
        self.cfg = prepare_config(self.raw, self.root, simulate=True)

    def test_epoch_default_and_invalid_configs(self):
        legacy = {k: v for k, v in self.raw.items() if k not in ("training_mode", "local_steps", "heterogeneity")}
        cfg = prepare_config(legacy, self.root, simulate=True)
        self.assertEqual(cfg["training_mode"], "epochs")
        self.assertFalse(cfg["heterogeneity"]["enabled"])
        for changes in ({"local_steps": None}, {"local_steps": 0}, {"local_steps": 1.1},
                        {"training_mode": "epochs"}, {"dataset": "speech_commands"},
                        {"reporting_fraction": 0.5}, {"strategy": "unknown"},
                        {"server_lr": 0}, {"server_momentum": -0.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                prepare_config({**self.raw, **changes}, self.root, simulate=True)

    def test_server_aggregation_settings_are_preserved(self):
        cfg = prepare_config({**self.raw, "strategy": "custom-fedavgm",
                              "server_lr": 1.0, "server_momentum": 0.0},
                             self.root, simulate=True)
        self.assertEqual(cfg["strategy"], "custom-fedavgm")
        self.assertEqual(cfg["server_lr"], 1.0)
        self.assertEqual(cfg["server_momentum"], 0.0)

    def test_distribution_constraints(self):
        for h in ({"reference_cpu": 0.9}, {"reference_client_id": 99}, {"mean": 0},
                  {"variance": -1}, {"variance": float("nan")}, {"distribution": "unknown"},
                  {"distribution": "homogeneous", "variance": 0.04},
                  {"distribution": "exponential", "mean": 2, "variance": 1},
                  {"initial_rounds": 3}, {"out_of_range": "clip"}, {"seed": 2**32}):
            with self.subTest(h=h), self.assertRaises(ValueError):
                prepare_heterogeneity(h, self.cfg)
        exp = prepare_heterogeneity({"distribution": "exponential", "mean": 2}, self.cfg)
        self.assertEqual(exp["variance"], 4)

    def test_normal_variance_and_seed_are_used_exactly(self):
        settings = prepare_heterogeneity({"mean": 1, "variance": 0.04, "seed": 123}, self.cfg)
        expected = np.random.default_rng(123).normal(1, 0.2, 3)
        result = sample_speed_factors(settings, ["2", "0", "1"])
        np.testing.assert_array_equal(list(result.values()), expected)
        exp = prepare_heterogeneity({"distribution": "exponential", "mean": 0.5, "seed": 123}, self.cfg)
        np.testing.assert_array_equal(list(sample_speed_factors(exp, ["0", "1", "2"]).values()),
                                      np.random.default_rng(123).exponential(0.5, 3))

    def test_plan_uses_reference_and_is_not_resampled(self):
        models = {cid: SpeedCpuModel(0.2 + i * 0.005, 0.8, 0.015, 0)
                  for i, cid in enumerate(self.cfg["client_ids"])}
        reference = models["0"].speed(0.5)
        path = self.root / "plan.json"
        plan = build_speed_plan(self.cfg, models, reference, path)
        self.assertTrue(plan["feasible"])
        self.assertEqual({row["requested_speed_steps_per_s"] for row in plan["clients"]}, {reference})
        self.assertNotEqual(plan["clients"][0]["cpu"], plan["clients"][2]["cpu"])
        with patch("warmup.heterogeneity.sample_speed_factors", side_effect=AssertionError("No resampling")):
            self.assertEqual(build_speed_plan(self.cfg, models, reference, path), plan)
        with self.assertRaises(ValueError):
            build_speed_plan(self.cfg, models, reference * 1.1, path)
        plan["clients"][0]["cpu"] = 0.9
        atomic_json(path, plan)
        with self.assertRaises(ValueError):
            read_plan(path)

    def test_unreachable_and_nonpositive_speeds_are_saved_not_clipped(self):
        models = dict.fromkeys(self.cfg["client_ids"], SpeedCpuModel(0.2, 0.8, 0.015, 0))
        path = self.root / "infeasible.json"
        with patch("warmup.heterogeneity.sample_speed_factors", return_value={"0": -1, "1": 100, "2": 1}):
            plan = build_speed_plan(self.cfg, models, models["0"].speed(0.5), path)
        self.assertFalse(plan["feasible"])
        self.assertIsNone(plan["clients"][0]["cpu"])
        self.assertIsNone(plan["clients"][1]["cpu"])
        self.assertEqual(plan["clients"][2]["cpu"], 0.5)
        self.assertEqual(read_plan(path), plan)


class StepPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cfg = prepare_config({"dataset": "femnist", "model": "cnn", "data_dir": "unused",
                                   "output_dir": "run", "training_mode": "steps", "local_steps": 20,
                                   "batch_size": 64, "heterogeneity": {"distribution": "normal", "variance": 0.04}},
                                  self.root, simulate=True)
        self.output = Path(self.cfg["output_dir"])
        self.jobs = []

    def runner(self, job):
        self.jobs.append(job)
        simulated_stage(job)

    def rows(self, name):
        with (self.output / name).open() as handle:
            return list(csv.DictReader(handle))

    def test_end_to_end_speed_target_selective_adjustment_and_logs(self):
        final = Calibration(self.cfg, self.runner).run()
        self.assertEqual([j["stage_id"].split("/")[0] for j in self.jobs[:5]],
                         ["scan_30", "scan_50", "scan_70", "scan_90", "heterogeneous_initial"])
        self.assertTrue(all(j["local_steps"] == 20 for j in self.jobs))
        profiles = self.rows("speed_profiles.csv")
        self.assertEqual(len(profiles), 12)
        self.assertTrue(all(int(row["n_samples"]) == 49 for row in profiles))
        reference = next(float(row["speed_steps_per_s"]) for row in profiles
                         if row["client_id"] == "0" and row["cpu"] == "0.5")
        self.assertEqual(read_plan(self.output / "heterogeneity_plan.json")["reference_speed_steps_per_s"], reference)
        target_record = validate_speed_target(self.output, self.cfg)
        target = target_record["theta_target_s"]
        measured = self.rows("heterogeneous_initial_measurements.csv")
        anchor = max(float(row["theta_s"]) for row in measured)
        expected = derive_pacer_target(anchor, len(self.cfg["client_ids"]))
        self.assertEqual({name: target_record[name] for name in expected}, expected)
        self.assertEqual(target_record["anchor_theta_s"], anchor)
        self.assertEqual(target_record["deadline_s"] * 2, round(target_record["deadline_s"] * 2))
        self.assertNotAlmostEqual(target, max(float(row["theta_s"]) for row in self.rows("profiles.csv") if row["cpu"] == "0.9"))
        history = self.rows("validation_history.csv")
        self.assertTrue(all(float(row["theta_target_s"]) == target and int(row["n_samples"]) == 29 for row in history))
        validation_jobs = self.jobs[5:]
        self.assertGreaterEqual(len(validation_jobs), 2)
        self.assertLessEqual(len(validation_jobs), 5)
        for current, following in zip(validation_jobs, validation_jobs[1:]):
            stage = current["stage_id"].split("/")[0]
            changes = json.loads((self.output / "stages" / stage / "adjustments.json").read_text())
            changed_ids = {row["client_id"] for row in changes}
            for a, b in zip(current["clients"], following["clients"]):
                if a["client_id"] not in changed_ids:
                    self.assertEqual(a, b)
        self.assertTrue(all(row["passed"] and row["converged"] for row in final))
        self.assertTrue(all(row["training_mode"] == "steps" and row["local_steps"] == 20 for row in final))
        for job in self.jobs:
            attempt = Path(job["output_dir"])
            self.assertTrue((attempt / "job.json").exists())
            self.assertTrue((attempt / "speeds.json").exists())
            self.assertTrue((attempt / "fits.json").exists())
            self.assertEqual(len((attempt / "client_0.jsonl").read_text().splitlines()), job["rounds"])
        self.assertFalse((self.output / "final_cpu_config.csv").exists())

    def test_resume_preserves_draws_target_and_raw_data(self):
        final = Calibration(self.cfg, self.runner).run()
        saved = {p: p.read_bytes() for p in [*self.output.rglob("*.jsonl"), self.output / "target.json",
                                            self.output / "heterogeneity_plan.json"]}
        with patch("warmup.heterogeneity.sample_speed_factors", side_effect=AssertionError("No draw")):
            resumed = Calibration(self.cfg, lambda job: self.fail("No launching on resume")).run(resume=True)
        self.assertEqual(final, resumed)
        self.assertTrue(all(path.read_bytes() == content for path, content in saved.items()))
        with self.assertRaises(ValueError):
            Calibration({**self.cfg, "local_steps": 21}, self.runner).run(resume=True)

    def test_initial_phase_uses_its_own_round_and_discard_counts(self):
        self.cfg["heterogeneity"].update(initial_rounds=12, initial_discard=2)
        Calibration(self.cfg, self.runner).run()
        rows = self.rows("heterogeneous_initial_measurements.csv")
        self.assertTrue(all(int(row["n_samples"]) == 10 for row in rows))
        complete = json.loads((self.output / "stages/heterogeneous_initial/complete.json").read_text())
        self.assertEqual((complete["rounds"], complete["discard"]), (12, 2))

    def test_last_measured_config_and_export_keep_fixed_step_launch(self):
        self.cfg.update(simulation=False, max_iterations=1)
        final = Calibration(self.cfg, self.runner).run()
        self.assertFalse(final[0]["converged"])
        self.assertEqual({r["client_id"]: r["cpu"] for r in final},
                         {r["client_id"]: r["cpu"] for r in self.jobs[-1]["clients"]})
        launcher = (self.output / "launch_final_clients.sh").read_text()
        for text in ("LOCAL_STEPS=20", "STEP_SEED=42", "ENABLE_CPU_AFFINITY=0", "BATCH_SIZE=64", "WARNING"):
            self.assertIn(text, launcher)
        self.assertNotIn("CPU_ONLY=1", launcher)
        control = ControlConfig.from_dict(json.loads((self.output / "pacer-control.json").read_text()))
        target = json.loads((self.output / "target.json").read_text())
        self.assertEqual(control.theta_target_s, target["theta_target_s"])
        self.assertEqual(control.deadline_s, target["deadline_s"])
        self.assertEqual(control.required_client_ids, tuple(self.cfg["client_ids"]))
        (self.output / "final_cpu_config.csv").unlink()
        with patch("warmup.controller.fit_logistic", side_effect=AssertionError("No refitting")):
            self.assertEqual(export_last(self.output), final)

    def test_corrupt_initial_target_cannot_be_exported(self):
        Calibration(self.cfg, self.runner).run()
        target = json.loads((self.output / "target.json").read_text())
        target["theta_target_s"] *= 1.1
        atomic_json(self.output / "target.json", target)
        with self.assertRaises(ValueError):
            validate_speed_target(self.output, self.cfg)
        with self.assertRaises(ValueError):
            export_last(self.output)

    def test_interrupted_initial_stage_keeps_plan_and_failed_attempt(self):
        def interrupted(job):
            simulated_stage(job)
            if job["stage_id"].startswith("heterogeneous_initial"):
                raise RuntimeError("Interrupted initial measurement")
        with self.assertRaises(RuntimeError):
            Calibration(self.cfg, interrupted).run()
        plan = (self.output / "heterogeneity_plan.json").read_bytes()
        source = self.output / "stages/heterogeneous_initial/attempt_001/client_0.jsonl"
        original = source.read_bytes()
        self.assertFalse((self.output / "target.json").exists())
        with patch("warmup.heterogeneity.sample_speed_factors", side_effect=AssertionError("No resampling")):
            Calibration(self.cfg, self.runner).run(resume=True)
        self.assertEqual((self.output / "heterogeneity_plan.json").read_bytes(), plan)
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue((self.output / "stages/heterogeneous_initial/attempt_002/client_0.jsonl").exists())

    def test_changed_initial_measurement_is_rejected_by_target_source(self):
        Calibration(self.cfg, self.runner).run()
        source = self.output / "stages/heterogeneous_initial/attempt_001/client_0.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows[2]["train_time_s"] *= 1.1
        rows[2]["metrics"]["steps_per_second"] = 20 / rows[2]["train_time_s"]
        source.write_text("\n".join(map(json.dumps, rows)) + "\n")
        with self.assertRaisesRegex(ValueError, "Fixed target differs"):
            validate_speed_target(self.output, self.cfg)

    def test_five_validation_batches_export_only_last_measured_cpu(self):
        self.cfg["simulation"] = False
        self.cfg["tolerance"] = 0.000001
        final = Calibration(self.cfg, self.runner).run()
        validation_jobs = [job for job in self.jobs if job["stage_id"].startswith("validate_")]
        self.assertEqual(len(validation_jobs), 5)
        self.assertTrue(all(row["validation_iteration"] == 5 and not row["converged"] for row in final))
        self.assertTrue((self.output / "final_cpu_config.csv").exists())
        self.assertEqual({row["client_id"]: row["cpu"] for row in final},
                         {row["client_id"]: row["cpu"] for row in validation_jobs[-1]["clients"]})
        self.assertFalse((self.output / "stages/validate_005/adjustments.json").exists())

    def test_wrong_actual_step_count_fails_without_final_output(self):
        def bad(job):
            simulated_stage(job)
            p = Path(job["output_dir"]) / "client_0.jsonl"
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            rows[1]["metrics"]["local_steps_used"] = 19
            p.write_text("\n".join(map(json.dumps, rows)) + "\n")
        with self.assertRaises(ValueError):
            Calibration(self.cfg, bad).run()
        self.assertFalse((self.output / "simulated_cpu_config.csv").exists())
        self.assertFalse((self.output / "stages/scan_30/complete.json").exists())

    def test_batch_workload_fingerprint(self):
        clients = [{"client_id": "0", "cpu": 0.5, "cpu_affinity": "-"}]
        key = batch_key(clients, 30, 1, self.cfg)
        for changed in ({"local_steps": 30}, {"batch_size": 32}, {"seed": 43}, {"training_mode": "epochs"}):
            self.assertNotEqual(key, batch_key(clients, 30, 1, {**self.cfg, **changed}))

    @patch("warmup.controller._stop_client_scopes")
    @patch("warmup.controller._wait_for_server")
    @patch("warmup.controller._port_is_open", return_value=False)
    @patch("warmup.controller.subprocess.Popen")
    def test_real_executor_forwards_workload_and_preserves_launch_architecture(self, popen, port, wait, stop):
        self.output.mkdir()
        job = {**self.cfg, "stage_id": "scan_30/attempt_001", "rounds": 50, "discard": 1,
               "cpu_map_csv": str(self.output / "requested_cpu_config.csv"),
               "server_cpu_affinity": "20-31,52-63", "server_csv_path": "logs/atest.csv",
               "clients": [{"client_id": "0", "cpu": 0.3, "cpu_affinity": "-"}]}
        simulated_stage(job)
        process = MagicMock()
        process.poll.return_value = 0
        process.returncode = 0
        popen.return_value = process
        external_stage(job)
        self.assertEqual(popen.call_count, 2)
        server, launcher = popen.call_args_list
        command = server.args[0]
        self.assertEqual(command[:3], ["taskset", "-c", "20-31,52-63"])
        self.assertIn("run_server", command)
        self.assertEqual(command[command.index("--local-steps") + 1], "20")
        self.assertEqual(command[command.index("--csv-path") + 1], "logs/atest.csv")
        self.assertTrue(launcher.args[0][1].endswith("launch_clients.sh"))
        env = launcher.kwargs["env"]
        self.assertEqual((env["LOCAL_STEPS"], env["STEP_SEED"], env["ENABLE_CPU_AFFINITY"]), ("20", "42", "0"))
        self.assertEqual(env["MEASUREMENT_RUN_ID"], job["stage_id"])
        self.assertEqual(env["LOCAL_EPOCHS"], str(self.cfg["epochs"]))
        self.assertNotIn("--local-only", launcher.args[0])
        stop.assert_called_once_with(["0"])


if __name__ == "__main__":
    unittest.main()
