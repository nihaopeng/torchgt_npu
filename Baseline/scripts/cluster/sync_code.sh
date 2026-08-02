#!/usr/bin/env bash
# Sync the whole repository to the same path on every node.
# Usage: bash sync_code.sh <1|2|4|8|16|hosts_file>
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${CLUSTER_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${BASELINE_ROOT}/.." && pwd)}"
REMOTE_REPO="${REMOTE_REPO:-${REPO_ROOT}}"
# shellcheck source=lib.sh
source "${CLUSTER_DIR}/lib.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 <1|2|4|8|16|hosts_file>" >&2
  exit 1
fi

SSH_KEY="${SSH_KEY:-/root/.ssh/id_ed25519}"
resolve_hosts_file "${CLUSTER_DIR}" "$1"
load_hosts "${HOSTS_FILE}"
default_ssh_opts

if [[ ! -f "${REPO_ROOT}/Baseline/main_sp_node_level.py" ]]; then
  echo "Not a Baseline repository: ${REPO_ROOT}" >&2
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
    --exclude '*_logs/' \
    --exclude '*_logs/**' \
    --exclude 'logs/' \
    --exclude 'logs/**' \
    --exclude '*.log' \
    --exclude '*.pid' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "${REPO_ROOT}/" "root@${HOST}:${REMOTE_REPO}/"
  ssh "${SSH_OPTS[@]}" "root@${HOST}" \
    "test -f '${REMOTE_REPO}/Baseline/main_sp_node_level.py'"
  echo "[${HOST}] OK"
done

echo "Done. BASELINE_ROOT=${REMOTE_REPO}/Baseline on each node."
