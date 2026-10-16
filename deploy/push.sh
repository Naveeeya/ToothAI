#!/usr/bin/env bash
# Build both images on this Mac (arm64, same as the VM), stream them to the VM and restart the stack.
#   deploy/push.sh <vm-ip> [--weights]     # --weights also syncs the model files (first deploy, or after retraining)
# Nothing is compiled on the VM. Weights travel over SSH only; they are never in git or the images.
set -euo pipefail
IP="${1:?usage: deploy/push.sh <vm-ip> [--weights]}"
KEY="${TOOTHAI_SSH_KEY:-$HOME/.ssh/toothai_oracle}"
# IPQoS=none: macOS ssh otherwise marks packets in a way some home routers choke on ("Result too large").
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o IPQoS=none -o ServerAliveInterval=15)
SSH=(ssh "${SSH_OPTS[@]}" "ubuntu@$IP")
VERSION="$(git rev-parse --short HEAD)"
SITE="${SITE_ADDRESS:-${IP//./-}.sslip.io}"
cd "$(git rev-parse --show-toplevel)"

if [[ -n "$(git status --porcelain -- app.py tooth_ai photo_screening docker deploy .streamlit static assets)" ]]; then
  echo "Uncommitted changes in the app: commit first so the image version matches git." >&2; exit 1
fi

echo "== build toothai-photo-api:$VERSION and toothai-ui:$VERSION (linux/arm64)"
docker build --platform linux/arm64 -f photo_screening/Dockerfile -t "toothai-photo-api:$VERSION" .
docker build --platform linux/arm64 -f docker/ui.Dockerfile -t "toothai-ui:$VERSION" .

# rsync with --partial and retries: a dropped connection resumes instead of starting over.
sync() {
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    rsync -a --partial --timeout=120 --exclude '._*' -e "ssh ${SSH_OPTS[*]}" "$@" && return 0
    echo "   transfer interrupted, retrying ($i)"; sleep 10
  done
  return 1
}

if [[ "${2:-}" == "--weights" ]]; then
  echo "== sync weights"
  for run in resnet50_9class_20k rtx4060_48k; do
    "${SSH[@]}" "mkdir -p /opt/toothai/output/$run"
    sync "output/$run/config.yaml" "output/$run/model_final.pth" "ubuntu@$IP:/opt/toothai/output/$run/"
  done
  sync --exclude '*.onnx' photo_screening/weights/ "ubuntu@$IP:/opt/toothai/weights/photo/"
fi

echo "== save images to a tar, send only what changed (rsync delta, compressed), load it on the VM"
# Uncompressed, under a fixed name: unchanged image layers are byte-identical between builds, so rsync's
# delta transfer sends little more than the changed code after the first deploy.
TMP="${TOOTHAI_DEPLOY_TMP:-.deploy}"
mkdir -p "$TMP"
TAR="$TMP/toothai-images.tar"
docker save "toothai-photo-api:$VERSION" "toothai-ui:$VERSION" -o "$TAR.tmp" && mv "$TAR.tmp" "$TAR"
rm -f "$TMP"/toothai-*.tar.gz
"${SSH[@]}" "mkdir -p /opt/toothai/images && rm -f /opt/toothai/images/*.tar.gz"
sync -z "$TAR" "ubuntu@$IP:/opt/toothai/images/"
"${SSH[@]}" "docker load -i /opt/toothai/images/toothai-images.tar"
echo "== start $VERSION at https://$SITE"
scp "${SSH_OPTS[@]}" deploy/compose.yml deploy/Caddyfile "ubuntu@$IP:/opt/toothai/"
"${SSH[@]}" "cd /opt/toothai && printf 'TOOTHAI_VERSION=%s\nSITE_ADDRESS=%s\n' '$VERSION' '$SITE' > .env \
  && docker compose -f compose.yml up -d --remove-orphans && docker image prune -af >/dev/null && docker compose ps"
echo "live: https://$SITE"
