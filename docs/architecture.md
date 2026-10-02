# Architecture

Current: pinned Hugging Face files stream through access-host memory into MinIO.
Every completed file is read back and SHA-256 checked. The access-host process is
single-instance and stores durable per-file manifests. No full dataset is staged
on either workstation or access-host disk.

Planned after GPU access: dedicated PVC cache → actual Arsenal training loop →
PVC checkpoints, logs and metrics → verified MinIO artifacts → validated inference
deployment. This part is not implemented or executed yet.
