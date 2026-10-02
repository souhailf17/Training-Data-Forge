"""Continue the verified four-rank pass with physical microbatch three.

The global batch is 384: 4 ranks * 3 samples * 32 accumulation steps.
The retained parent marker is verified on PVC and S3 before a real-update,
checkpoint/replay and kernel-execution gate is run. The gate is rolled back
before durable continuation so it never consumes training coverage twice.
"""
import json
import math
import os
import random
import shutil
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["TRAIN_MODE"] = "train"

import numpy as np
import torch
import torch.distributed as dist

import train_optimized as base


RANKS = int(os.environ.get("TRAIN_RANKS", "4"))
assert RANKS == 4, "This continuation is approved and tested for exactly four DDP ranks"
MICROBATCH = 3
TOTAL = 3_683_080
BATCH = 384
ACCUMULATION = BATCH // (RANKS * MICROBATCH)
RUN = base.SOURCE / "training-fast-gpu4-m3-from7878-v1"
PREFIX = "training-fast-gpu4-m3-from7878-v1/"
PARENT_MARKER = base.SOURCE / "training-fast-gpu4-m1-from6343-v1/latest.json"
PARENT = json.loads(PARENT_MARKER.read_text())
assert PARENT["global_step"] == 7878, PARENT
assert PARENT["full_samples"] == 1_196_288, PARENT
assert Path(PARENT["path"]).name == "checkpoint-7878.pt", PARENT

# The imported implementation deliberately keeps its run globals explicit.
# Redirect those globals to this isolated continuation before constructing it.
base.RUN = RUN
base.RANKS = RANKS
base.BATCH = BATCH
base.TOTAL = TOTAL


def upload(path: Path):
    key = PREFIX + str(path.relative_to(RUN))
    sha = base.digest(path)
    base.connect().upload_file(str(path), base.BUCKET, key)
    base.verify_s3(key, sha)
    return {"path": str(path), "key": key, "sha256": sha}


base.upload = upload


def collate(rows, pad_token_id):
    result = {}
    width = max(row["input_ids"].shape[1] for row in rows)
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids", "token_type_ids"):
            fill = -100 if key == "labels" else pad_token_id if key == "input_ids" else 0
            values = [
                torch.nn.functional.pad(value, (0, width - value.shape[1]), value=fill)
                for value in values
            ]
        elif key not in ("pixel_values", "image_grid_thw"):
            raise ValueError("Unvalidated processor batch field: " + key)
        result[key] = torch.cat(values, dim=0)
    return base.arsenal_ocr.OCRBatch(result)


class BatchedStream:
    def __init__(self, inner, batch_size=MICROBATCH):
        self.inner = inner
        self.batch_size = batch_size

    @property
    def processor(self):
        return self.inner.processor

    @property
    def count(self):
        return self.inner.count

    @property
    def cursor(self):
        return self.inner.cursor

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state):
        self.inner.load_state_dict(state)

    def encode(self, *args, **kwargs):
        return self.inner.encode(*args, **kwargs)

    def __iter__(self):
        iterator = iter(self.inner)
        while True:
            rows = []
            for _ in range(self.batch_size):
                try:
                    rows.append(next(iterator))
                except StopIteration:
                    break
            if not rows:
                return
            yield collate(rows, self.inner.processor.tokenizer.pad_token_id)


def spawn_m3(self):
    return base.OrderedPrefetch(BatchedStream(base.FastStream(self.model_path)), depth=4)


base.FastData.spawn = spawn_m3


class ResumeTrainer(base.FastTrainer):
    def restore_full(self, entry):
        """Restore the verified four-rank parent exactly."""
        self.close_loader()
        path = Path(entry["path"])
        assert base.digest(path) == entry["sha256"]
        if dist.get_rank() == 0:
            base.verify_s3(entry["key"], entry["sha256"])
        dist.barrier()
        state = torch.load(path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, parameter in self.models["model"].named_parameters():
                if parameter.requires_grad:
                    value = state["adapter"].get(
                        name, state["adapter"].get(name.removeprefix("module."))
                    )
                    assert value is not None, name
                    parameter.copy_(value)
        self.optimizer.load_state_dict(state["optimizer"])
        self.clock.load_state_dict(state["clock"])
        self.dataloader.load_state_dict(state["loader"])

        rank = dist.get_rank()
        rank_rng = state.get("rank_rng", [])
        if rank < len(rank_rng):
            rank_state = rank_rng[rank]
            random.setstate(rank_state["python_rng"])
            np.random.set_state(rank_state["numpy_rng"])
            torch.set_rng_state(rank_state["torch_rng"])
            cuda_rng = rank_state["cuda_rng"]
            torch.cuda.set_rng_state(cuda_rng[0] if isinstance(cuda_rng, list) else cuda_rng)
        else:
            raise AssertionError(("Missing per-rank RNG state", rank, len(rank_rng)))

        self.lr_scheduler.set_lr(step=state["scheduler_step"])
        self.consumed = state["full_samples"]
        self.source_samples = state["source_samples"]
        assert self.dataloader.count == self.consumed
        self.iterator = iter(self.dataloader)
        self.sync_check()

    def update(self, count=BATCH):
        world = dist.get_world_size()
        assert world == RANKS
        physical_microbatch = MICROBATCH
        if count < BATCH:
            assert count % world == 0
            physical_microbatch = count // world
            assert 0 < physical_microbatch < MICROBATCH
            state = self.dataloader.state_dict()
            self.close_loader()
            self.dataloader.inner.batch_size = physical_microbatch
            self.dataloader.load_state_dict(state)
            self.iterator = iter(self.dataloader)
        samples_per_microstep = world * physical_microbatch
        assert count % samples_per_microstep == 0, (count, samples_per_microstep)
        self.config.gradient_accumulation.count = count // samples_per_microstep
        assert self.config.gradient_accumulation.count == ACCUMULATION or count < BATCH
        self.loss_sum = 0.0
        self.loss_count = 0
        self.step(self.iterator)
        self.clock.step_index += 1
        self.metrics.clear()
        self.consumed += count
        stats = torch.tensor([self.loss_sum, self.loss_count], device=self.device, dtype=torch.float64)
        dist.all_reduce(stats)
        return float(stats[0] / stats[1])

    def preflight_resume(self):
        state = torch.load(PARENT["path"], map_location="cpu", weights_only=False)
        assert state["world_size"] == RANKS
        assert state["full_samples"] == PARENT["full_samples"]
        assert state["loader"]["count"] == PARENT["full_samples"]
        assert PARENT["full_samples"] % RANKS == 0

        self.restore_full(PARENT)
        baseline = self.evaluate()
        assert math.isfinite(baseline)

        original = self.adapter_state()
        first_loss = self.update(BATCH)
        assert math.isfinite(first_loss)
        assert any(not torch.equal(original[name], value) for name, value in self.adapter_state().items())
        del original

        checkpoint = self.save_full("preflight-4gpu-m3", publish=False)
        expected_loss = self.update(BATCH)
        expected = self.adapter_state()
        self.restore_full(checkpoint)
        repeated_loss = self.update(BATCH)
        error = max(float((expected[name] - value).abs().max()) for name, value in self.adapter_state().items())
        loss_error = abs(expected_loss - repeated_loss)
        del expected
        assert error < 1e-5 and loss_error < 1e-4, (error, expected_loss, repeated_loss)
        self.sync_check()
        assert base.CALLS.get("causal_conv1d_fn", 0) > 0
        assert base.CALLS.get("chunk_gated_delta_rule", 0) > 0

        proofs = [None] * RANKS
        dist.all_gather_object(proofs, {
            "rank": dist.get_rank(),
            "resume_error": error,
            "loss_error": loss_error,
            "first_loss": first_loss,
            "kernel_calls": base.CALLS.copy(),
            "gpu": torch.cuda.get_device_name(),
            "max_memory_allocated": torch.cuda.max_memory_allocated(),
            "max_memory_reserved": torch.cuda.max_memory_reserved(),
        })

        # Roll the gate back to the durable parent before processing new data.
        self.restore_full(PARENT)
        assert self.consumed == PARENT["full_samples"]
        if dist.get_rank() == 0:
            result = {
                "status": "passed",
                "parent": PARENT,
                "baseline_eval_loss": baseline,
                "world_size": RANKS,
                "global_batch": BATCH,
                "microbatch": MICROBATCH,
                "accumulation": ACCUMULATION,
                "ranks": proofs,
                "gates": [
                    "verified_parent_pvc_s3",
                    "four_rank_cursor_restore",
                    "finite_gradients",
                    "parameter_update",
                    "rank_synchronization",
                    "exact_four_rank_resume",
                    "pvc_s3_verified_checkpoint",
                    "kernel_execution",
                ],
                "coverage": f"The gate is rolled back to full_samples={PARENT['full_samples']} before continuation.",
            }
            base.write(RUN / "preflight.json", result)
            upload(RUN / "preflight.json")
            base.write(RUN / "best.json", dict(PARENT, eval_loss=baseline))
            upload(RUN / "best.json")
            print("GPU4_M3_PREFLIGHT_PASSED", json.dumps(result), flush=True)
        dist.barrier()
        return baseline

    def run_full(self):
        self.iterator = None
        best = self.preflight_resume()
        start_samples = self.consumed
        started = time.time()
        last_save = started
        history = []
        bad = 0
        reason = "complete"
        updates = 0

        while self.consumed < TOTAL:
            began = time.time()
            loss = self.update(min(BATCH, TOTAL - self.consumed))
            updates += 1
            history.append(loss)
            entry = {
                "timestamp": time.time(),
                "state": "running",
                "global_step": self.clock.step_index,
                "resume_updates": updates,
                "full_samples": self.consumed,
                "total_samples": TOTAL,
                "train_loss": loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "update_seconds": time.time() - began,
                "world_size": RANKS,
                "global_batch": BATCH,
                "microbatch": MICROBATCH,
                "accumulation": self.config.gradient_accumulation.count,
            }
            stop = None
            if len(history) >= 13 and all(value > 3 * statistics.median(history[-13:-3]) for value in history[-3:]):
                stop = "divergent_loss"
            if (len(history) >= 20 and statistics.pstdev(history[-10:]) > statistics.mean(history[-10:])
                    and statistics.mean(history[-10:]) > 1.5 * statistics.median(history[-20:-10])):
                stop = "unstable_loss"
            stop = self.decision(stop or ("requested_stop" if (RUN / "STOP").exists() else None))
            if self.decision(shutil.disk_usage(RUN).free < 4 * 1024**3):
                stop = "disk_reserve_reached_no_files_deleted"

            save = self.decision(
                updates == 1 or updates % 250 == 0 or time.time() - last_save >= 3600
                or bool(stop) or self.consumed == TOTAL
            )
            if save:
                score = self.evaluate()
                entry["eval_loss"] = score
                self.sync_check()
                marker = self.save_full(str(self.clock.step_index))
                last_save = time.time()
                if score < best:
                    best = score
                    if dist.get_rank() == 0:
                        base.write(RUN / "best.json", dict(marker, eval_loss=best))
                        upload(RUN / "best.json")
                bad = bad + 1 if score > best * 1.2 and score - best > 0.05 else 0
                if bad >= 2:
                    stop = "validation_degradation"

            elapsed = time.time() - started
            processed = self.consumed - start_samples
            rate = processed / elapsed if elapsed else 0.0
            entry.update(
                elapsed_seconds=elapsed,
                examples_per_second=rate,
                estimated_remaining_seconds=(TOTAL - self.consumed) / rate if rate else None,
                state="stopped" if stop else "running",
                stop_reason=stop,
            )
            if dist.get_rank() == 0:
                with (RUN / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(entry) + "\n")
                base.write(RUN / "status.json", entry)
                if save or updates % 10 == 0:
                    upload(RUN / "metrics.jsonl")
                    upload(RUN / "status.json")
                print("GPU4_M3_TRAIN_METRIC", json.dumps(entry), flush=True)
            if stop:
                reason = stop
                break

        self.close_loader()
        if dist.get_rank() == 0:
            result = {
                "state": "complete" if self.consumed == TOTAL else "stopped",
                "reason": reason,
                "full_samples": self.consumed,
                "resume_updates": updates,
                "global_step": self.clock.step_index,
                "elapsed_seconds": time.time() - started,
                "best_eval_loss": best,
            }
            base.write(RUN / "result.json", result)
            upload(RUN / "result.json")
        dist.barrier()
        self.clock.stop()


def main():
    assert BATCH == RANKS * MICROBATCH * ACCUMULATION
    remainder = (TOTAL - PARENT["full_samples"]) % BATCH
    assert remainder == 8 and remainder % RANKS == 0
    assert not RUN.exists(), "Inspect existing continuation; never overwrite or restart automatically"
    parent_path = Path(PARENT["path"])
    assert parent_path.is_file()
    assert base.digest(parent_path) == PARENT["sha256"]
    if dist.is_available() and dist.is_initialized():
        raise AssertionError("Trainer launcher must initialize DDP")
    base.verify_s3(PARENT["key"], PARENT["sha256"])

    benchmark_key = "optimization/benchmark-prefetch-v3/result.json"
    benchmark = json.loads(base.connect().get_object(Bucket=base.BUCKET, Key=benchmark_key)["Body"].read())
    assert benchmark["status"] == "passed" and len(benchmark["ranks"]) == 4
    assert all(rank["resume_error"] < 1e-5 for rank in benchmark["ranks"])
    assert all(rank["kernel_calls"].get("chunk_gated_delta_rule", 0) > 0 for rank in benchmark["ranks"])
    assert shutil.disk_usage(base.SOURCE).free > 10 * 1024**3

    RUN.mkdir()
    (RUN / "logs").mkdir()
    base.write(RUN / "started.json", {
        "time": time.time(),
        "parent": PARENT,
        "coverage": f"Continue the same deterministic full pass from {PARENT['full_samples']} to {TOTAL}.",
        "node": "scheduler-selected",
        "world_size": RANKS,
        "global_batch": BATCH,
        "microbatch": MICROBATCH,
        "accumulation": ACCUMULATION,
        "kernels": {"fla-core": "0.5.2", "causal-conv1d": "1.7.0", "triton": base.triton.__version__},
    })
    upload(RUN / "started.json")

    config = base.FastConfig(
        run_directory=RUN / "arsenal",
        model=base.OCRModel(model_path=str(base.SOURCE / "model")),
        data=base.FastData(model_path=str(base.SOURCE / "model"), batch_size=MICROBATCH),
        optimizer=base.AdamConfig(),
        lr_scheduler=base.CosineSchedulerConfig(
            init_lr=1e-6, max_lr=1e-4, min_lr_ratio=.1, max_steps=460385,
            use_warmup=True, warmup_steps=500, alpha=1.,
        ),
        distributed=base.DistributedConfig(
            device_type="cuda", num_process_per_node=RANKS,
            hsdp=base.HsdpConfig(replicate_size=RANKS, shard_size=1), use_local_loss=True,
        ),
        compile=base.CompileConfig(enabled=False),
        dtype=torch.bfloat16,
    )
    ResumeTrainer(config).train()


if __name__ == "__main__":
    main()
