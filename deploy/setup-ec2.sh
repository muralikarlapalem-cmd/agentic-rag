#!/usr/bin/env bash
# Provision and (re)start the app on an EC2 instance.
#
#   sudo ./deploy/setup-ec2.sh [--ssm-prefix /agentic-rag] [--domain rag.example.com]
#
# Supports Ubuntu 22.04/24.04 and Amazon Linux 2023. Safe to re-run.
set -euo pipefail

SSM_PREFIX=""; DOMAIN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssm-prefix) SSM_PREFIX="$2"; shift 2 ;;
    --domain)     DOMAIN="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }
cd "$(dirname "$0")/.."

. /etc/os-release
echo "==> Detected OS: ${PRETTY_NAME}"

install_docker() {
  if docker compose version >/dev/null 2>&1; then echo "==> Docker + compose already installed"; return; fi
  case "$ID" in
    ubuntu|debian)
      apt-get update -y
      apt-get install -y docker.io docker-compose-v2 curl ;;
    amzn)
      dnf install -y docker curl
      arch="$(uname -m)"
      mkdir -p /usr/local/lib/docker/cli-plugins
      curl -fsSL "https://github.com/docker/compose/releases/download/v2.32.4/docker-compose-linux-${arch}" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
      chmod +x /usr/local/lib/docker/cli-plugins/docker-compose ;;
    *) echo "Unsupported OS '$ID'. Install Docker + the compose plugin manually, then re-run." >&2; exit 1 ;;
  esac
  systemctl enable --now docker
  docker compose version
}

fetch_secrets() {
  command -v aws >/dev/null 2>&1 || {
    case "$ID" in
      ubuntu|debian) snap install aws-cli --classic ;;
      amzn) dnf install -y awscli ;;
    esac
  }
  echo "==> Reading secrets from SSM Parameter Store at ${SSM_PREFIX}"
  umask 077
  local tmp; tmp="$(mktemp)"
  aws ssm get-parameters-by-path --path "$SSM_PREFIX" --with-decryption \
      --query 'Parameters[].[Name,Value]' --output text |
    while IFS=$'\t' read -r name value; do printf '%s=%s\n' "${name##*/}" "$value"; done > "$tmp"
  [[ -s "$tmp" ]] || { echo "No parameters found under ${SSM_PREFIX} (check IAM role and region)." >&2; exit 1; }
  mv "$tmp" .env
}

install_docker

if [[ -n "$SSM_PREFIX" ]]; then
  fetch_secrets
elif [[ ! -f .env ]]; then
  echo "No .env and no --ssm-prefix given. Create .env from .env.example (chmod 600) or use SSM." >&2
  exit 1
fi
chmod 600 .env

grep -q '^OPENAI_API_KEY=.\+' .env || { echo ".env is missing OPENAI_API_KEY" >&2; exit 1; }
grep -q '^APP_PASSWORD=.\+'   .env || { echo ".env is missing APP_PASSWORD (required for a hosted deployment)" >&2; exit 1; }

PROFILE=()
if [[ -n "$DOMAIN" ]]; then
  grep -v '^DOMAIN=' .env > .env.tmp || true; echo "DOMAIN=${DOMAIN}" >> .env.tmp; mv .env.tmp .env; chmod 600 .env
  PROFILE=(--profile https)
fi

echo "==> Building and starting"
docker compose "${PROFILE[@]}" up -d --build

echo "==> Waiting for health check"
for _ in $(seq 1 40); do
  status="$(docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q app)" 2>/dev/null || true)"
  [[ "$status" == "healthy" ]] && { echo "==> App is healthy"; break; }
  sleep 3
done
[[ "${status:-}" == "healthy" ]] || { echo "App did not become healthy. Logs:" >&2; docker compose logs --tail=50 app >&2; exit 1; }

if [[ -n "$DOMAIN" ]]; then echo "==> Live at https://${DOMAIN}  (first request may take a few seconds while the certificate is issued)"
else echo "==> Running on 127.0.0.1:8501. Reach it with:  ssh -L 8501:localhost:8501 <user>@<instance-ip>"; fi
