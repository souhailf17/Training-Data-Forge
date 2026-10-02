> Current update: the user selected priority 0 without a PriorityClass. The replacement probe succeeded. Leave all node placement to OpenShift. See scheduling.md; earlier PriorityClass blockers below are historical.

# Troubleshooting

- SSH banner timeout on the access host: resolved on retry; do not restart remote services.
- An alternate access host had no configured kubeconfig; use the approved project access host.
- Hugging Face HTTP 401: use the access host's approved credential; public metadata alone is insufficient.
- PowerShell TLS failure: Python urllib worked with certificate verification enabled.
- RBAC forbidden for nodes/all namespaces/PriorityClasses: obtain read access or the approved class name; do not escalate privileges.
- Probe admission rejects Never: use a PriorityClass configured for Never; do not remove the protection merely to make scheduling succeed.
- Transfer mismatch: keep the failure log, retry the current source object, and never report it as verified.
- Pending GPU: wait for genuinely available resources; do not delete or scale other workloads.

The initial failed runtime probe remains suspended for inspection. Only resources
labelled project=q35ocr-20260921-p560 were created or changed by this task.
