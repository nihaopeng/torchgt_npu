#!/usr/bin/env bash
# Rsync this git repo to every host in hosts.N (same path on all nodes).
#
# Usage:
#   bash sync_code.sh <1|2|4|8|16|hosts_file>
#
# Env:
#   REPO_ROOT=<repo>               # local repo (contains NeutronGT/ code dir)
#   REMOTE_REPO=<remote_repo>      # same path on workers
#   SSH_KEY=/root/.ssh/id_ed25519
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${CLUSTER_DIR}/lib.sh"

HOSTS_ARG="${1:-}"
if [[ -z "${HOSTS_ARG}" ]]; then
  echo "Usage: bash $0 <1|2|4|8|16|hosts_file>" >&2
  exit 1
fi

SSH_KEY="${SSH_KEY:-/root/.ssh/id_ed25519}"
REPO_ROOT="${REPO_ROOT:-$(cd "${CLUSTER_DIR}/../../.." && pwd)}"
REMOTE_REPO="${REMOTE_REPO:-${REPO_ROOT}}"

resolve_hosts_file "${CLUSTER_DIR}" "${HOSTS_ARG}"
load_hosts "${HOSTS_FILE}"
default_ssh_opts

if [[ ! -f "${REPO_ROOT}/NeutronGT/main_sp_node_level_ppr.py" ]]; then
  echo "Not a NeutronGT repo: ${REPO_ROOT}" >&2
  exit 1
fi

BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "sync repo branch=${BRANCH} commit=${COMMIT}"
echo "${REPO_ROOT}/ -> root@<host>:${REMOTE_REPO}/ (${NNODES} nodes)"

for HOST in "${HOSTS[@]}"; do
  echo "[${HOST}] rsync"
  ssh "${SSH_OPTS[@]}" "root@${HOST}" "mkdir -p '${REMOTE_REPO}'"
  rsync -az --delete \
    -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.git/' \
    --exclude 'NeutronGT_logs/' \
    --exclude 'NeutronGT/NeutronGT_logs/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'NeutronGT/wrap_neutron_main.py' \
    "${REPO_ROOT}/" "root@${HOST}:${REMOTE_REPO}/"
  echo "[${HOST}] OK"
done

echo "Done. NEUTRON_ROOT=${REMOTE_REPO}/NeutronGT on each node."
