#!/usr/bin/env bash
# Collect all node logs for the latest Baseline run.
# Usage: bash collect_logs.sh <1|2|4|8|16|hosts_file> <arxiv|reddit|products|papers>
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${CLUSTER_DIR}/../.." && pwd)"
# shellcheck source=lib.sh
source "${CLUSTER_DIR}/lib.sh"

if [[ $# -ne 2 ]]; then
  echo "Usage: bash $0 <1|2|4|8|16|hosts_file> <arxiv|reddit|products|papers>" >&2
  exit 1
fi

HOSTS_ARG="$1"
DATASET_ALIAS="$(normalize_dataset_alias "$2")"
LOG_ROOT="${LOG_ROOT:-${BASELINE_ROOT}/TorchGT_logs/cluster}"
SSH_KEY="${SSH_KEY:-/root/.ssh/id_ed25519}"

resolve_hosts_file "${CLUSTER_DIR}" "${HOSTS_ARG}"
load_hosts "${HOSTS_FILE}"
default_ssh_opts

STATE_FILE="${LOG_ROOT}/state_baseline_${DATASET_ALIAS}_${NNODES}card.env"
if [[ ! -f "${STATE_FILE}" ]]; then
  echo "Missing ${STATE_FILE} (run launch.sh first)" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${STATE_FILE}"

LOG_DIR="${LOG_ROOT}/${DATASET_ALIAS}/${NNODES}card"
mkdir -p "${LOG_DIR}"
for RANK in "${!HOSTS[@]}"; do
  HOST="${HOSTS[$RANK]}"
  for SUFFIX in log pid; do
    REMOTE_FILE="${LOG_DIR}/node${RANK}_${RUN_TAG}.${SUFFIX}"
    scp "${SSH_OPTS[@]}" "root@${HOST}:${REMOTE_FILE}" "${LOG_DIR}/" 2>/dev/null || true
  done
done
echo "Collected under ${LOG_DIR}/ (run_tag=${RUN_TAG})"
