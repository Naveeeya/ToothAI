#!/usr/bin/env bash
# Create the Always Free ARM VM, retrying while Oracle reports "out of capacity" (common for A1 in India).
#   TENANCY=ocid1.tenancy... SUBNET=ocid1.subnet... deploy/oracle-launch.sh
# Needs the OCI CLI configured (~/.oci/config). Fixed to Always Free limits: VM.Standard.A1.Flex,
# 2 OCPU, 12 GB, one 100 GB boot volume. Stops at the first success, or if a `toothai` instance exists.
set -uo pipefail
OCI="${OCI:-oci}"
TENANCY="${TENANCY:?set TENANCY to the tenancy OCID}"
SUBNET="${SUBNET:?set SUBNET to the public subnet OCID}"
NAME="${NAME:-toothai}"
KEY="${KEY:-$HOME/.ssh/toothai_oracle.pub}"
WAIT="${WAIT:-60}"

AD="$($OCI iam availability-domain list -c "$TENANCY" --query 'data[0].name' --raw-output 2>/dev/null)"
IMAGE="$($OCI compute image list -c "$TENANCY" --operating-system "Canonical Ubuntu" --operating-system-version 24.04 \
  --shape VM.Standard.A1.Flex --sort-by TIMECREATED --sort-order DESC --limit 1 --query 'data[0].id' --raw-output 2>/dev/null)"
echo "$(date '+%F %T') AD=$AD image=${IMAGE: -12} name=$NAME"

attempt=0
while true; do
  attempt=$((attempt + 1))
  # Any live A1 instance (also one created from the console) uses up the free allowance: stop, never add a second.
  existing="$($OCI compute instance list -c "$TENANCY" \
    --query 'data[?shape==`VM.Standard.A1.Flex` && "lifecycle-state"!=`TERMINATED` && "lifecycle-state"!=`TERMINATING`].id | [0]' \
    --raw-output 2>/dev/null)"
  if [[ -n "$existing" && "$existing" != "null" ]]; then
    ID="$existing"; echo "$(date '+%F %T') an A1 instance already exists; not creating another"; break
  fi
  out="$($OCI compute instance launch -c "$TENANCY" --availability-domain "$AD" --display-name "$NAME" \
    --shape VM.Standard.A1.Flex --shape-config '{"ocpus": 2, "memoryInGBs": 12}' \
    --image-id "$IMAGE" --subnet-id "$SUBNET" --assign-public-ip true --boot-volume-size-in-gbs 100 \
    --ssh-authorized-keys-file "$KEY" --query 'data.id' --raw-output 2>&1)"
  if [[ "$out" == ocid1.instance.* ]]; then
    ID="$out"; echo "$(date '+%F %T') created on attempt $attempt"; break
  fi
  if grep -qi "capacity" <<<"$out"; then
    (( attempt % 10 == 1 )) && echo "$(date '+%F %T') attempt $attempt: out of capacity, retrying every ${WAIT}s"
    sleep "$WAIT"; continue
  fi
  if grep -qiE "TooManyRequests|429" <<<"$out"; then
    echo "$(date '+%F %T') rate limited; waiting 5 min"; sleep 300; continue
  fi
  # Client-side network trouble (DNS, dropped connection, timeout): not Oracle's answer, so retry.
  if grep -qiE "ConnectionError|Max retries|NameResolution|timed out|RequestException|\"target_service\": \"CLI\"" <<<"$out"; then
    echo "$(date '+%F %T') network error on this machine; retrying in 2 min"; sleep 120; continue
  fi
  echo "$(date '+%F %T') unexpected error, stopping:"; echo "$out" | grep -v SyntaxWarning | head -12; exit 1
done

$OCI compute instance get --instance-id "$ID" --wait-for-state RUNNING --max-wait-seconds 900 >/dev/null 2>&1
IP="$($OCI compute instance list-vnics --instance-id "$ID" --query 'data[0]."public-ip"' --raw-output 2>/dev/null)"
echo "$(date '+%F %T') RUNNING  public IP: $IP"
echo "ssh -i ${KEY%.pub} ubuntu@$IP"
