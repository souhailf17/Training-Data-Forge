# Four-GPU migration

Update: distributed smoke v5 passed all four ranks with exactly zero resume
error. Enabling deterministic algorithms in every spawned rank resolved the
earlier numerical discrepancy. Job `q35ocr-20260921-p560-train4-v1` was launched
from verified checkpoint step750 (5992 consumed samples). The original
`q35ocr-20260921-p560-train-v1` is suspended. OpenShift selected worker9; no
worker was specified in the submitted job.

Monitor: `oc -n mtrna-ns1 logs -f job/q35ocr-20260921-p560-train4-v1`.

The user authorized accelerating training using multiple GPUs. A scheduler-chosen
probe confirmed four NVIDIA H200 NVL devices. No node name, node selector,
affinity, or PriorityClass was submitted. Priority remains zero.

`distributed_ocr.py` uses the actual Arsenal launcher and training step with
PyTorch DistributedDataParallel. Four ranks share synchronized adapters and
consume disjoint samples. Two microbatches per rank preserve effective batch8.
The original one-GPU run remains active until a distributed smoke test passes.

The node hosting four GPUs cannot connect directly to the external MinIO IP.
A project-owned, bucket-restricted TCP relay provides in-cluster connectivity.
Its Service sends traffic only to replicas whose upstream connectivity check
passes. It forwards existing S3 authentication unchanged. No unrelated relay,
network policy, service or workload is modified.

The smoke test checks finite gradients, parameter updates, rank synchronization,
restored update consistency, and S3 artifact checksums. Distributed loader tests
exercise exact disjointness and continuation from a saved global cursor.

Migration verifies the existing local and S3 checkpoint before suspending only
the project's one-GPU job. Unsaved samples are replayed from the last durable
checkpoint. The same96Gi PVC is then attached by OpenShift to the new job.
After migration, four-rank checkpoints gather per-rank RNG state, use the highest
consumed global cursor, and persist the synchronized optimizer/adapter state.

Performance must be measured after migration; GPU count alone is not a reliable
speedup guarantee. Existing Qwen linear-attention optimized kernels are absent
from the runtime, which currently uses the Transformers PyTorch fallback.
