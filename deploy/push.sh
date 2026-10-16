#!/usr/bin/env bash
# Build both images on this Mac (arm64, same as the VM), stream them to the VM and restart the stack.
#   deploy/push.sh <vm-ip> [--weights]     # --weights also syncs the model files (first deploy, or after retraining)
# Nothing is compiled on the VM. Weights travel over SSH only; they are never in git or the images.
set -euo pipefail
IP="${1:?usage: deploy/push.sh <vm-ip> [--weights]}"
KEY="${TOOTHAI_SSH_KEY:-$HOME/.ssh/toothai_oracle}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$IP")
VERSION="$(git rev-parse --short HEAD)"
SITE="${SITE_ADDRESS:-${IP//./-}.sslip.io}"
cd "$(git rev-parse --show-toplevel)"

if [[ -n "$(git status --porcelain -- app.py tooth_ai photo_screening docker deploy .streamlit static assets)" ]]; then
  echo "Uncommitted changes in the app: commit first so the image version matches git." >&2; exit 1
fi

echo "== build toothai-photo-api:$VERSION and toothai-ui:$VERSION (linux/arm64)"
docker build --platform linux/arm64 -f photo_screening/Dockerfile -t "toothai-photo-api:$VERSION" .
docker build --platform linux/arm64 -f docker/ui.Dockerfile -t "toothai-ui:$VERSION" .

if [[ "${2:-}" == "--weights" ]]; then
  echo "== sync weights"
  RSYNC=(rsync -a --progress --exclude '._*' -e "ssh -i $KEY")
  "${RSYNC[@]}" output/resnet50_9class_20k output/rtx4060_48k "ubuntu@$IP:/opt/toothai/output/" \
    --include '*/' --include 'config.yaml' --include 'model_final.pth' --exclude '*'
  "${RSYNC[@]}" photo_screening/weights/ "ubuntu@$IP:/opt/toothai/weights/photo/" --exclude '*.onnx'
fi

echo "== stream images (compressed) and load them on the VM"
docker save "toothai-photo-api:$VERSION" "toothai-ui:$VERSION" | gzip -1 | "${SSH[@]}" 'gunzip | docker load'

echo "== start $VERSION at https://$SITE"
scp -i "$KEY" deploy/compose.yml deploy/Caddyfile "ubuntu@$IP:/opt/toothai/"
"${SSH[@]}" "cd /opt/toothai && printf 'TOOTHAI_VERSION=%s\nSITE_ADDRESS=%s\n' '$VERSION' '$SITE' > .env \
  && docker compose -f compose.yml up -d --remove-orphans && docker image prune -af >/dev/null && docker compose ps"
echo "live: https://$SITE"
