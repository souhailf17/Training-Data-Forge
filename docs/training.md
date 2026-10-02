> Current update: the user selected priority 0 without a PriorityClass. The replacement probe succeeded. Leave all node placement to OpenShift. See scheduling.md; earlier PriorityClass blockers below are historical.

# Training — full job launched after successful Arsenal smoke

Current implementation: `scripts/arsenal_ocr.py`, `scripts/train_full.py`.
Smoke job `q35ocr-20260921-p560-smoke-v2` passed every gate, including identical
resumed parameter updates (maximum error 0), finite gradients, assistant-only
labels and PVC/S3 checkpoint round trip. H200 peak allocation: 48,147,557,376 bytes.

Full job: `q35ocr-20260921-p560-train-v1`. One epoch across 3,683,080 train
examples, deterministic shard/row shuffle, BF16 frozen model, float32 rank16
alpha32 language adapters (29,097,984 parameters), no dropout, AdamW LR1e-4,
weight decay .01, warmup500, cosine decay, microbatch1, accumulation8, clip1.
Images fit within 1024x1024 preserving aspect ratio; exact labels are preserved.
Sequences above8192tokens cause explicit failure, never silent truncation.

Validation monitors64distinct groups across32shards at baseline and every1000
updates. Final held-out OCR evaluation remains a separate completion gate.
Checkpoints after the first update and every250steps include adapters, optimizer,
loader, clock and RNG; all are uploaded with full checksum read-back. Three recent
local checkpoints remain on PVC, while S3 retains all copies.

PVC root: `/work/q35ocr-20260921-p560`; logs in `logs/training.log`,
`logs/training-status.json`, `logs/metrics.jsonl`, `logs/validation.jsonl`.
Resume pointer: `checkpoints/latest.json`. Never launch a second live trainer.
Use `oc -n mtrna-ns1 logs -f job/q35ocr-20260921-p560-train-v1` to monitor.

The submitted job has no node selector or affinity. Admission adds the namespace
tenant selector `atlasx.ma/tenant=mtrna` to the actual pod. No specific worker is
requested. The scheduler selected worker7 for this run. Pod priority is 0.
The SDK clock is one-based; status.step records the next SDK step after an update.

The following describes the historical initial probe only:

The GPU/runtime probe was rejected before pod creation because the cluster
requires an approved non-preempting PriorityClass. It is suspended. No training
method or GPU allocation has been finalized. The PVC budget provisionally targets
LoRA; full tuning has not been sized or approved by a memory measurement.

Inspect installed Arsenal version, its trainer lifecycle, optimizer, checkpoint
and loader-state interfaces before writing the training integration. Existing
Gemma scripts are reference evidence only. Existing Qwen scripts use ms-swift,
which is not sufficient evidence of Arsenal training integration.

Full training must follow the smoke gates in README.md, including backward,
parameter updates, complete checkpoint reload, data pairing, and durable storage.
