#!/usr/bin/env python
"""Wrapper around NeutronGT main for multi-node without shared filesystem.

The current implementation syncs via local disk under dataset_dir:
  - ppr_temp/ppr_shard_*.pt          (struct_info.gather_ppr_shards)
  - runtime_sync/preprocess_done_*   (main_sp_node_level_ppr markers)
  - window_state_cache/*             (window_state bundles + .done)

This wrapper scp's those files to NEUTRONG_PEER_HOSTS after each write.
Does not modify the NeutronGT git tree beyond dropping this overlay file.

Usage (same argv as main_sp_node_level_ppr.py):
  NEUTRONG_PEER_HOSTS=172.18.43.123 \\
  NEUTRONG_SSH_KEY=/root/.ssh/id_ed25519 \\
  python -m torch.distributed.run ... wrap_neutron_main.py ...
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _peers() -> list[str]:
    return [h.strip() for h in os.environ.get("NEUTRONG_PEER_HOSTS", "").split(",") if h.strip()]


def _ssh_key() -> str:
    return os.environ.get("NEUTRONG_SSH_KEY", "/root/.ssh/id_ed25519")


def _scp_to_peers(path: str | os.PathLike) -> None:
    peers = _peers()
    if not peers:
        return
    path = os.fspath(path)
    if not os.path.isfile(path):
        return

    ssh_key = _ssh_key()
    remote_dir = os.path.dirname(path) or "."
    for peer in peers:
        mkdir_cmd = [
            "ssh", "-i", ssh_key,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            f"root@{peer}",
            f"mkdir -p {remote_dir}",
        ]
        scp_cmd = [
            "scp", "-i", ssh_key,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            path,
            f"root@{peer}:{path}",
        ]
        try:
            subprocess.run(mkdir_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(scp_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[wrap_neutron] synced {path} -> {peer}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[wrap_neutron] WARN: failed to sync {path} -> {peer}: {exc}", flush=True)


def _should_sync_path(path: str) -> bool:
    return any(
        key in path
        for key in ("window_state_cache", "ppr_temp", "runtime_sync")
    )


def _install_torch_save_hook() -> None:
    """Fallback for direct torch.save into sync dirs (non-atomic writers)."""
    if not _peers():
        return
    import torch

    orig_save = torch.save

    def save_and_sync(obj, f, *args, **kwargs):
        ret = orig_save(obj, f, *args, **kwargs)
        path = f if isinstance(f, (str, bytes, os.PathLike)) else getattr(f, "name", None)
        if path is None:
            return ret
        path = os.fspath(path)
        # Skip tmp files from atomic writers; those are synced after os.replace.
        if ".tmp." in os.path.basename(path):
            return ret
        if _should_sync_path(path):
            _scp_to_peers(path)
        return ret

    torch.save = save_and_sync  # type: ignore[assignment]
    print(f"[wrap_neutron] torch.save hook enabled; peers={_peers()}", flush=True)


def _patch_atomic_writers() -> None:
    """Patch atomic save/touch helpers so final paths are scp'd after replace."""
    if not _peers():
        return

    import core.node_level_pipeline.struct_info as struct_info
    import core.node_level_pipeline.window_state as window_state
    import main_sp_node_level_ppr as neutron_main

    def _wrap_atomic(orig):
        def atomic_and_sync(obj, path, *args, **kwargs):
            orig(obj, path, *args, **kwargs)
            if _should_sync_path(os.fspath(path)):
                _scp_to_peers(path)
        return atomic_and_sync

    def _wrap_touch(orig):
        def touch_and_sync(path, *args, **kwargs):
            orig(path, *args, **kwargs)
            p = os.fspath(path)
            if _should_sync_path(p):
                _scp_to_peers(p)
        return touch_and_sync

    struct_info._atomic_torch_save = _wrap_atomic(struct_info._atomic_torch_save)
    window_state._atomic_torch_save = _wrap_atomic(window_state._atomic_torch_save)
    window_state._touch_done = _wrap_touch(window_state._touch_done)
    neutron_main._touch_marker = _wrap_touch(neutron_main._touch_marker)
    print(
        "[wrap_neutron] patched atomic writers: ppr_temp + runtime_sync + window_state_cache",
        flush=True,
    )


def main() -> None:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    _install_torch_save_hook()
    _patch_atomic_writers()

    import main_sp_node_level_ppr as neutron_main

    sys.argv[0] = os.path.join(cwd, "main_sp_node_level_ppr.py")
    neutron_main.main()


if __name__ == "__main__":
    main()
