#!/usr/bin/env bash
set -euo pipefail

AXERA_DIR="${AXERA_DIR:-/mnt/axera}"
for f in "${AXERA_DIR}"/logs/*.pid; do
  [ -f "$f" ] || continue
  pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop] $(basename "$f" .pid) ($pid)"
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$f"
done
echo "全部服务已停止"
