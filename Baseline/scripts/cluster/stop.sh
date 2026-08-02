#!/usr/bin/env bash
# Stop only Baseline main_sp_node_level.py jobs.
# Usage: bash stop.sh <1|2|4|8|16|hosts_file>
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

for HOST in "${HOSTS[@]}"; do
  echo "Stopping Baseline on ${HOST}..."
  ssh "${SSH_OPTS[@]}" "root@${HOST}" \
    "pkill -f 'torch\.distributed\.run.*[ /][m]ain_sp_node_level\.py([ ]|$)' || true;
     pkill -f '(^|[ /])[m]ain_sp_node_level\.py([ ]|$)' || true" \
    || true
done
echo "Done."
