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
RSH="ssh $(printf '%q ' "${SSH_OPTS[@]}")"   # rsync -e is re-split by a shell: quote each option
VERSION="$(git rev-parse --short HEAD)"
SITE="${SITE_ADDRESS:-${IP//./-}.sslip.io}"
cd "$(git rev-parse --show-toplevel)"

# A deploy uploads ~1 GB the first time. On a metered network (phone hotspot, cellular: macOS marks it
# "expensive") that can cost money, so refuse unless explicitly allowed.
if [[ "$(uname)" == "Darwin" ]] && [[ -z "${TOOTHAI_ALLOW_METERED:-}" ]]; then
  iface="$(route -n get "$IP" 2>/dev/null | awk '/interface/{print $2}')"
  if [[ -n "$iface" ]] && ipconfig getsummary "$iface" 2>/dev/null | grep -q "IsExpensive : TRUE"; then
    echo "This Mac is on a metered network ($iface is marked expensive, e.g. a phone hotspot)." >&2
    echo "Switch to Wi-Fi/broadband, or set TOOTHAI_ALLOW_METERED=1 to upload anyway." >&2
    exit 1
  fi
fi

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
    rsync -a --chmod=D755,F644 --partial --timeout=120 --exclude '._*' -e "$RSH" "$@" && return 0
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

echo "== save images, upload them in 20 MB pieces (each resumable), reassemble and verify on the VM"
# Phone hotspots drop long uploads; small pieces each finish between drops, and rsync skips pieces that
# are already there (same name and content), so an interrupted or repeated deploy resumes where it stopped.
TMP="${TOOTHAI_DEPLOY_TMP:-.deploy}"
mkdir -p "$TMP/parts"
TAR="$TMP/toothai-images.tar"
docker save "toothai-photo-api:$VERSION" "toothai-ui:$VERSION" -o "$TAR.tmp" && mv "$TAR.tmp" "$TAR"
if [[ -n "${TOOTHAI_STOP_DOCKER:-}" ]]; then colima stop >/dev/null 2>&1 || true; fi   # free memory for the upload
rm -f "$TMP"/toothai-*.tar.gz "$TMP"/parts/part-*
split -b 20m "$TAR" "$TMP/parts/part-"
SUM="$(shasum -a 256 "$TAR" | awk '{print $1}')"
"${SSH[@]}" "mkdir -p /opt/toothai/images/parts && rm -f /opt/toothai/images/*.tar.gz /opt/toothai/images/toothai-images.tar"
n=0; total=$(ls "$TMP"/parts/part-* | wc -l | tr -d ' ')
for part in "$TMP"/parts/part-*; do
  n=$((n + 1)); sync "$part" "ubuntu@$IP:/opt/toothai/images/parts/" || { echo "piece $n failed" >&2; exit 1; }
  (( n % 10 == 0 )) && echo "   $n/$total pieces"
done
"${SSH[@]}" "cd /opt/toothai/images && ls parts/part-* | sort | xargs cat > toothai-images.tar \
  && [ \"\$(sha256sum toothai-images.tar | cut -d' ' -f1)\" = $SUM ] && echo 'checksum ok' \
  && docker load -i toothai-images.tar && rm -rf parts toothai-images.tar"
echo "== start $VERSION at https://$SITE"
scp "${SSH_OPTS[@]}" deploy/compose.yml deploy/Caddyfile "ubuntu@$IP:/opt/toothai/"
"${SSH[@]}" "cd /opt/toothai && printf 'TOOTHAI_VERSION=%s\nSITE_ADDRESS=%s\n' '$VERSION' '$SITE' > .env \
  && docker compose -f compose.yml up -d --remove-orphans && docker image prune -af >/dev/null && docker compose ps"
echo "live: https://$SITE"
