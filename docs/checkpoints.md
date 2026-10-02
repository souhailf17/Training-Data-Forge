# Checkpoints — pending training integration

No checkpoints exist. Planned active path is /checkpoints on the dedicated PVC;
remote prefix is training/checkpoints/ in the project bucket.

Keep latest, best validation and final checkpoints. A resumable checkpoint must
include adapter/model weights, optimizer, scheduler, global step, epoch, loader
cursor and RNG state. Only publish completion markers after all files are saved
and S3 read-back hashes match. Never sync files while the trainer is modifying them.
Exact Arsenal resume invocation remains unverified and is intentionally not invented.
