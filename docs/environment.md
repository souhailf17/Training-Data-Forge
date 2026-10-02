> Current update: the user selected priority 0 without a PriorityClass. The replacement probe succeeded. Leave all node placement to OpenShift. See scheduling.md; earlier PriorityClass blockers below are historical.

# Environment

The original environment used a remote access host and the OpenShift CLI. Host,
identity and credential details are intentionally omitted from this repository.
The project namespace used during the documented run was `mtrna-ns1`.
Access-host root filesystem had approximately 26 GiB free at discovery; streaming
avoids consuming this with model weights or raw shards. Workstation had roughly
380 GB free; recheck before final model download.

No ResourceQuota or LimitRange objects were returned in this namespace. Cluster
node capacity and other namespaces are not visible to this account. Inventory is
recorded in evidence/cluster-inventory.json. The approved non-preempting
PriorityClass is still unknown.
