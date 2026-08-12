#!/usr/bin/env bash
# 用 hf-mirror 的 hfd.sh 多线程下载 HuggingFace 模型（国内镜像优先）。
#
# 用法:
#   axera/scripts/download_hf.sh <REPO_ID> [hfd 参数...]
# 示例:
#   axera/scripts/download_hf.sh AXERA-TECH/bge-m3 --local-dir axera/models/bge-m3 -x 8
set -euo pipefail

HFD_URL="${HFD_URL:-https://hf-mirror.com/hfd/hfd.sh}"
HFD_CACHE="${HFD_CACHE:-${HOME}/.cache/magnetar/hfd.sh}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ "$#" -lt 1 ]; then
    echo "用法: $0 <REPO_ID> [hfd 参数...]" >&2
    exit 2
fi

mkdir -p "$(dirname "$HFD_CACHE")"
if [ ! -f "$HFD_CACHE" ]; then
    echo "[download_hf] 获取 hfd.sh: ${HFD_URL}"
    curl -fsSL --retry 2 --connect-timeout 15 -o "$HFD_CACHE" "$HFD_URL"
fi
chmod +x "$HFD_CACHE"

if ! command -v aria2c &>/dev/null && [[ " $* " != *" --tool "* ]]; then
    echo "[download_hf] 未找到 aria2c，回退 wget（建议: apt install aria2）" >&2
    set -- "$@" --tool wget
fi

export HF_ENDPOINT
exec "$HFD_CACHE" "$@"
