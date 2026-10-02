# Security and credentials

Do not commit OpenShift tokens, S3 access keys, Hugging Face tokens, passwords, kubeconfigs, model weights, checkpoints or raw OCR data.

The code expects credentials to be injected at runtime. Common variable names are:

- `AWS_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `PROJECT_BUCKET`

On OpenShift, store these values in a Secret and expose them to the pod through `envFrom` or individual `secretKeyRef` entries. Rotate a credential immediately if it is ever committed.

