#!/usr/bin/env bash
# Stop NeutronGT cluster jobs on hosts.
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

JIT_LOCK_GLOB='/tmp/torch_extensions_*/**/lock'
for HOST in "${HOSTS[@]}"; do
  echo "Stopping on ${HOST}..."
  ssh "${SSH_OPTS[@]}" "root@${HOST}" \
    "pkill -f torch.distributed.run || true; pkill -f main_sp_node_level_ppr || true; pkill -f wrap_neutron_main || true; rm -f ${JIT_LOCK_GLOB} || true" \
    || true
done
echo "Done."
