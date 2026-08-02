#!/usr/bin/env bash
# Wait for a Baseline cluster run, then collect its logs.
# Usage: bash wait.sh <1|2|4|8|16|hosts_file> <arxiv|reddit|products|papers>
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
  echo "Missing state file: ${STATE_FILE} (run launch.sh first)" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${STATE_FILE}"

LOG_DIR="${LOG_ROOT}/${DATASET_ALIAS}/${NNODES}card"
echo "Waiting for ${RUN_TAG} on ${NNODES} nodes..."
while true; do
  RUNNING=0
  for RANK in "${!HOSTS[@]}"; do
    HOST="${HOSTS[$RANK]}"
    PID_FILE="${LOG_DIR}/node${RANK}_${RUN_TAG}.pid"
    if ssh "${SSH_OPTS[@]}" "root@${HOST}" \
      "test -f '${PID_FILE}' && kill -0 \$(cat '${PID_FILE}') 2>/dev/null"; then
      RUNNING=$((RUNNING + 1))
    fi
  done
  echo "[$(date '+%F %T')] ${RUNNING}/${NNODES} nodes running"
  [[ "${RUNNING}" -eq 0 ]] && break
  sleep 30
done

bash "${CLUSTER_DIR}/collect_logs.sh" "${HOSTS_ARG}" "${DATASET_ALIAS}"
echo "Done. Logs: ${LOG_DIR}"
