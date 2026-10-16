# Deploying ToothAI on Oracle Cloud Always Free

One small ARM server runs everything: the Streamlit app (OPG analysis + Photo screening), the photo
screening API and Caddy (automatic HTTPS). Docker images are built on an arm64 Mac and streamed to the
server over SSH; nothing is compiled on the server. Model weights travel over SSH only.

```
browser ──HTTPS──▶ caddy :443 ──▶ ui :8501 (Streamlit, Detectron2) ──▶ photo-api :8001 (internal only)
```

## Cost: what is free and what could charge you

- An Oracle **Free Tier account that is never upgraded to "Pay As You Go" cannot be charged.** The card at
  sign-up is identity verification (a small temporary hold, reversed). The 30-day trial uses $300 of
  credit, not your card; when it ends, anything that isn't Always Free is stopped.
- This setup uses Always Free resources only (checked against Oracle's docs, Oct 2026): one
  `VM.Standard.A1.Flex` with **2 OCPU and 12 GB** (the whole free ARM allowance; never provision more),
  one boot volume (100 GB of the free 200 GB), and well under the free 10 TB/month of outbound data.
- Always Free resources exist only in your **home region**, chosen once at sign-up.
- Oracle may **reclaim idle** free VMs (7 days with CPU, network and memory all below 20 %). The two apps
  keep about 5 GB of the 12 GB in memory, so the VM shouldn't count as idle. Reclaiming deletes the VM; it
  does not charge you.
- Add a guard anyway: *Billing & Cost Management → Budgets → Create budget*, amount 1 USD, alert at 100 %.

## 1. Create the VM (once, in the Oracle console)

1. Sign up at <https://www.oracle.com/cloud/free/>. Home region: **India West (Mumbai)** or **India South
   (Hyderabad)**. Do not upgrade the account.
2. Menu **☰ → Compute → Instances → Create instance**. Name it `toothai`.
3. **Image and shape**
   - *Image*: **Change image → Ubuntu → Canonical Ubuntu 24.04** (the aarch64 build is picked for ARM).
   - *Shape*: **Change shape → Ampere → VM.Standard.A1.Flex**, **2 OCPUs**, **12 GB** memory.
4. **Networking**: create a new virtual cloud network with a **public subnet**; keep **Assign a public IPv4
   address** on.
5. **Add SSH keys → Paste public keys**, and paste the contents of `~/.ssh/toothai_oracle.pub`
   (create it with `ssh-keygen -t ed25519 -f ~/.ssh/toothai_oracle` if it doesn't exist).
6. **Boot volume**: *Specify a custom boot volume size*, 100 GB.
7. **Create.** If it says *Out of capacity for shape VM.Standard.A1.Flex*, pick another availability domain
   or retry later (free ARM capacity comes and goes). Don't upgrade to get capacity.
   To retry automatically instead: configure the OCI CLI (*Profile → My profile → API keys → Add API key*,
   save the key and config in `~/.oci/`), create the network with the VCN Wizard, then run
   `TENANCY=<tenancy OCID> SUBNET=<public subnet OCID> deploy/oracle-launch.sh`. It retries every minute,
   is fixed to 2 OCPU / 12 GB, and stops once any A1 instance exists.
8. Open ports 80 and 443: on the instance page, **Primary VNIC → Subnet → Security → Default Security List
   → Add Ingress Rules**: source `0.0.0.0/0`, TCP, destination port `80`; repeat for `443`.
9. Note the **Public IP address** on the instance page.

## 2. Open the VM (SSH)

```bash
ssh -i ~/.ssh/toothai_oracle ubuntu@<public-ip>
```

The user is `ubuntu`. The first connection asks to confirm the host key; type `yes`. Log out with `exit`.

## 3. Prepare it (once, from the Mac, in the repository root)

```bash
ssh -i ~/.ssh/toothai_oracle ubuntu@<public-ip> 'bash -s' < deploy/setup-vm.sh
```

It installs updates, automatic security upgrades, fail2ban, Docker and 4 GB of swap; opens 80/443 in the
VM's firewall (Oracle's Ubuntu image blocks them by default, separately from the security list); makes SSH
key-only; and creates `/opt/toothai/`.

## 4. Deploy

Colima (or Docker Desktop) must be running on the Mac with at least 4 GB of memory.

```bash
deploy/push.sh <public-ip> --weights     # first time: also copies the model weights (~930 MB)
deploy/push.sh <public-ip>               # every update after that
```

It refuses to run with uncommitted app changes, so the deployed version always matches a git commit, and
on a metered network (a phone hotspot or cellular link, which macOS marks "expensive"), because the first
deploy uploads about 1 GB of images (later deploys send only what changed). Set `TOOTHAI_ALLOW_METERED=1` to
override. The
site comes up at `https://<ip-with-dashes>.sslip.io` (free DNS for the IP; Caddy gets the certificate).
To use your own domain, point it at the IP and run `SITE_ADDRESS=your.domain deploy/push.sh <ip>`.

## 5. Operate

```bash
ssh -i ~/.ssh/toothai_oracle ubuntu@<ip> 'cd /opt/toothai && docker compose ps'          # status
ssh -i ~/.ssh/toothai_oracle ubuntu@<ip> 'cd /opt/toothai && docker compose logs --tail 100 ui'
ssh -i ~/.ssh/toothai_oracle ubuntu@<ip> 'cd /opt/toothai && docker compose restart'
ssh -i ~/.ssh/toothai_oracle ubuntu@<ip> 'docker stats --no-stream'                      # memory per service
```

Containers restart on their own after a crash or a reboot. Logs rotate at 10 MB × 3 per service.

## Measured speed (2 CPU cores, Docker on an M1; the Ampere cores may be somewhat slower)

| Task | Time |
| --- | --- |
| OPG, Full chart (two models, two passes each) | 19–22 s |
| Photo screening, 4–5 photos (AlphaDent at 960 px) | 48–68 s |
| Photo PDF | under 1 s |

One analysis runs at a time per service; a second visitor is asked to try again in a minute.

## Troubleshooting

- **Site doesn't load**: check both the security list (step 1.8) and the VM firewall (`sudo iptables -L
  INPUT -n | grep -E "80|443"`), then `docker compose logs caddy`.
- **Certificate errors right after the first deploy**: Caddy needs ports 80 and 443 reachable to get it;
  wait a minute and reload.
- **A service restarts in a loop**: `docker compose logs <service>`; `docker stats` for memory.
