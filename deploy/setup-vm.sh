#!/usr/bin/env bash
# One-time setup of the Oracle Always Free VM (Ubuntu 24.04, arm64). Run as the default `ubuntu` user:
#   ssh -i ~/.ssh/toothai_oracle ubuntu@<ip> 'bash -s' < deploy/setup-vm.sh
# Idempotent: safe to run again.
set -euo pipefail

echo "== system updates and automatic security upgrades"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y unattended-upgrades fail2ban docker.io docker-compose-v2 \
    iptables-persistent
sudo dpkg-reconfigure -f noninteractive unattended-upgrades

echo "== 4 GB swap (headroom while both apps load their models)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "== firewall: Oracle's Ubuntu image rejects everything but SSH in iptables; open 80 and 443"
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 5 -p tcp --dport "$port" -m state --state NEW -j ACCEPT
done
sudo iptables -C INPUT -p udp --dport 443 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 5 -p udp --dport 443 -j ACCEPT
sudo netfilter-persistent save

echo "== SSH: keys only"
sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl reload ssh 2>/dev/null || sudo systemctl restart ssh.socket 2>/dev/null || true   # 24.04: socket-activated

echo "== Docker"
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo "== app folders"
sudo mkdir -p /opt/toothai/output /opt/toothai/weights/photo
sudo chown -R "$USER":"$USER" /opt/toothai

echo "done. Log out and back in once so the docker group applies."
