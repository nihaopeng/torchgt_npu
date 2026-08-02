#!/usr/bin/env bash
# Shared helpers for Baseline/scripts/cluster.
# shellcheck shell=bash

resolve_hosts_file() {
  local cluster_dir="$1"
  local arg="$2"
  if [[ "${arg}" =~ ^(1|2|4|8|16)$ ]]; then
    HOSTS_FILE="${cluster_dir}/hosts.${arg}"
  elif [[ -f "${arg}" ]]; then
    HOSTS_FILE="${arg}"
  elif [[ -f "${cluster_dir}/${arg}" ]]; then
    HOSTS_FILE="${cluster_dir}/${arg}"
  else
    echo "Hosts not found: ${arg} (use 1|2|4|8|16 or a hosts file path)" >&2
    return 1
  fi
}

load_hosts() {
  local hosts_file="$1"
  mapfile -t HOSTS < <(grep -vE '^[[:space:]]*(#|$)' "${hosts_file}" | sed 's/[[:space:]]//g')
  NNODES="${#HOSTS[@]}"
  if [[ "${NNODES}" -lt 1 ]]; then
    echo "No hosts in ${hosts_file}" >&2
    return 1
  fi
}

resolve_dataset() {
  local alias="$1"
  local default_seq_len
  case "${alias}" in
    arxiv|ogbn-arxiv)
      DATASET="ogbn-arxiv"
      default_seq_len=256000
      ;;
    reddit)
      DATASET="reddit"
      default_seq_len=32000
      ;;
    products|ogbn-products)
      DATASET="ogbn-products"
      default_seq_len=256000
      ;;
    papers|papers100M|ogbn-papers100M)
      DATASET="ogbn-papers100M"
      default_seq_len=256000
      ;;
    *)
      echo "Unknown dataset: ${alias} (arxiv|reddit|products|papers)" >&2
      return 1
      ;;
  esac
  SEQ_LEN="${SEQ_LEN:-${default_seq_len}}"
}

normalize_dataset_alias() {
  case "$1" in
    ogbn-arxiv) echo "arxiv" ;;
    ogbn-products) echo "products" ;;
    papers100M|ogbn-papers100M) echo "papers" ;;
    *) echo "$1" ;;
  esac
}

default_ssh_opts() {
  local key="${SSH_KEY:-/root/.ssh/id_ed25519}"
  SSH_OPTS=(-i "${key}" -o BatchMode=yes -o StrictHostKeyChecking=no
            -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null
            -o LogLevel=ERROR -o ConnectTimeout=10)
}
