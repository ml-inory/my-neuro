#!/usr/bin/env bash
# AX650 一键部署 my-neuro axera 后端（ASR / TTS / RAG / BERT / LLM）。
#
# 用法：
#   在 x86 开发机（本仓库已克隆）：
#     bash axera/deploy/install_board.sh --remote root@<BOARD_IP>
#   直接在 AX650 板端（仓库已放到板上）：
#     bash axera/deploy/install_board.sh
#
# 依赖：root 权限；板端需要能访问 NFS（默认 10.122.86.219:/）并挂载到 /mnt。
set -Eeuo pipefail

REMOTE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

if [ -n "${REMOTE}" ]; then
  # ---- 宿主机模式：把仓库同步到 NFS 共享目录，再远程执行 ----
  NFS_HOST="${AXERA_NFS_HOST:-10.122.86.219}"
  echo "[deploy] 同步仓库到 ${NFS_HOST}:/data/shared/axera/my-neuro ..."
  DEST="/data/shared/axera/my-neuro"
  mkdir -p "$DEST"
  # 用 rsync 或 cp -r 同步（排除 .git 以减小体积，但保留以便 submodule 初始化）
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --exclude 'axera/models/Qwen2.5-1.5B-Instruct' --exclude 'axera/models/bge-m3' \
      ./ "$DEST/"
  else
    cp -a . "$DEST"
  fi
  echo "[deploy] 远程执行板端安装 ..."
  if command -v sshpass >/dev/null 2>&1 && ! ssh -o BatchMode=yes -o ConnectTimeout=5 "root@${REMOTE#*@}" true 2>/dev/null; then
    SSHPASS="${AXERA_BOARD_PASSWORD:-123456}" sshpass -e ssh -o StrictHostKeyChecking=no "$REMOTE" \
      "cd /mnt/axera/my-neuro && bash axera/deploy/install_board.sh"
  else
    ssh -o StrictHostKeyChecking=no "$REMOTE" \
      "cd /mnt/axera/my-neuro && bash axera/deploy/install_board.sh"
  fi
  exit $?
fi

# ==================== 板端模式 ====================
[ "$(id -u)" -eq 0 ] || { echo "请以 root 运行"; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AXERA_DIR="${AXERA_DIR:-/mnt/axera}"
NFS_HOST="${AXERA_NFS_HOST:-10.122.86.219}"
ENV_DIR="${AXERA_DIR}/env"
MODELS_DIR="${AXERA_DIR}/models"
LOG_DIR="${AXERA_DIR}/logs"

echo "==> 1/8 准备 NFS 存储（${NFS_HOST}:/ -> /mnt）"
if ! mountpoint -q /mnt; then
  mkdir -p /mnt
  mount -t nfs4 "${NFS_HOST}:/" /mnt -o nolock || mount -t nfs "${NFS_HOST}:/data/shared" /mnt -o nolock
fi
mkdir -p "${AXERA_DIR}"/{models,logs,bin,bsp}

echo "==> 2/8 校验板端基础环境（无需 apt，全部 Python 依赖装入 /mnt）"
command -v python3 pip3 git curl wget unzip >/dev/null || {
  echo "缺少基础命令，请先安装: python3 git curl wget unzip" >&2; exit 1; }
python3 -c "import soundfile" 2>/dev/null || echo "[warn] soundfile 缺失（稍后 pip 安装）"

echo "==> 3/8 拉取依赖仓库（sensevoice.axera / melotts.axera）"
cd "${REPO_ROOT}/axera"
if [ -d "${AXERA_DIR}/deps/sensevoice.axera" ] && [ -d "${AXERA_DIR}/deps/melotts.axera" ]; then
  # NFS 上已由开发机预置（含模型），直接链接
  [ -L deps ] || rm -rf deps
  ln -sfn "${AXERA_DIR}/deps" deps
else
  [ -L deps ] || rm -rf deps
  mkdir -p deps
  git clone --depth 1 https://github.com/ml-inory/sensevoice.axera.git deps/sensevoice.axera || \
  git clone --depth 1 https://gh-proxy.com/https://github.com/ml-inory/sensevoice.axera.git deps/sensevoice.axera
  git clone --depth 1 https://github.com/ml-inory/melotts.axera.git deps/melotts.axera || \
  git clone --depth 1 https://gh-proxy.com/https://github.com/ml-inory/melotts.axera.git deps/melotts.axera
fi

echo "==> 4/8 安装 Python 依赖到 ${AXERA_DIR}/pylib（--target，不占根分区）"
PY_TARGET="${AXERA_DIR}/pylib/site-packages"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export TMPDIR="${AXERA_DIR}/tmp" PIP_CACHE_DIR="${AXERA_DIR}/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
mkdir -p "$PY_TARGET"
python3 -m pip install --no-input --target "$PY_TARGET" --upgrade pip setuptools wheel 2>/dev/null || true
# axengine 用 NFS 上的本地 wheel（板端直连 github 常超时），其余依赖走 aliyun 镜像
grep -v '^axengine' "${REPO_ROOT}/axera/requirements.txt" > "${AXERA_DIR}/req_axera.txt"
if [ -d "${AXERA_DIR}/wheels" ] && ls "${AXERA_DIR}/wheels"/*.whl >/dev/null 2>&1; then
  # 开发机已预置 aarch64 wheels（pip download --platform manylinux2014_aarch64），离线安装
  REQ="${AXERA_DIR}/wheels/requirements-axera.txt"
  [ -f "$REQ" ] || REQ="${AXERA_DIR}/req_axera.txt"
  python3 -m pip install --no-input --no-index --find-links "${AXERA_DIR}/wheels" --target "$PY_TARGET" -r "$REQ" \
    || echo "[warn] 离线安装有缺失，尝试在线补充（需板端能访问镜像）"
else
  python3 -m pip install --no-input --target "$PY_TARGET" -r "${AXERA_DIR}/req_axera.txt"
fi
if [ -f "${AXERA_DIR}/axengine-0.1.3-py3-none-any.whl" ]; then
  python3 -m pip install --no-input --target "$PY_TARGET" "${AXERA_DIR}/axengine-0.1.3-py3-none-any.whl"
elif [ -f "${AXERA_DIR}/axengine.whl" ]; then
  python3 -m pip install --no-input --target "$PY_TARGET" "${AXERA_DIR}/axengine.whl"
fi
# 依赖仓库的额外 Python 依赖（MeloTTS 中文路径必需子集；mecab 等日文依赖失败不阻塞）
python3 -m pip install --no-input --target "$PY_TARGET" \
  pypinyin jieba cn2an g2p_en g2pkk jamo num2words "librosa==0.9.1" || true
python3 -m pip install --no-input --target "$PY_TARGET" \
  -r "${REPO_ROOT}/axera/deps/sensevoice.axera/python/requirements.txt" || true
# MeloTTS 需要 nltk_data
[ -d "${HOME}/nltk_data" ] || cp -rf "${REPO_ROOT}/axera/deps/melotts.axera/nltk_data" "${HOME}/" 2>/dev/null || true

echo "==> 5/8 下载模型（写入 ${MODELS_DIR}，较大请耐心等待）"
if [ ! -f "${MODELS_DIR}/vad/silero_vad.onnx" ] || [ ! -f "${MODELS_DIR}/bert/omni_fn_bert.axmodel" ] || [ ! -f "${MODELS_DIR}/bge-m3/model/bge-m3_u16_npu3.axmodel" ]; then
  AXERA_MODELS_DIR="${MODELS_DIR}" bash "${REPO_ROOT}/axera/models/download_models.sh"
else
  echo "[deploy] 模型已由开发机预置（vad/bert/bge-m3/qwen），跳过下载"
fi

echo "==> 6/8 部署 Her.axera 云端 LLM 网关（OpenAI 兼容 /v1/chat/completions）"
HER_DIR="${AXERA_DIR}/deps/Her.axera"
if [ ! -d "${HER_DIR}" ]; then
  git clone --depth 1 https://github.com/ml-inory/Her.axera.git "${HER_DIR}" || \
  git clone --depth 1 https://gh-proxy.com/https://github.com/ml-inory/Her.axera.git "${HER_DIR}"
fi
if [ ! -f "${HER_DIR}/backend/.env" ]; then
  cat > "${HER_DIR}/backend/.env" <<EOF
API_PREFIX=/v1
DEFAULT_LLM_PROVIDER=deepseek
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-}
DEEPSEEK_MODEL=deepseek-chat
ENABLE_OPENAI_COMPAT=${ENABLE_OPENAI_COMPAT:-false}
OPENAI_COMPAT_API_BASE=${OPENAI_COMPAT_API_BASE:-}
OPENAI_COMPAT_API_KEY=${OPENAI_COMPAT_API_KEY:-}
OPENAI_COMPAT_MODEL=${OPENAI_COMPAT_MODEL:-gpt-4o}
# ASR/TTS 由 Her.axera 提供（ax_asr / ax_tts，NPU）
ENABLE_AX_ASR=true
AX_ASR_MODEL_TYPE=sensevoice
AX_ASR_MODEL_PATH=/mnt/axera/deps/sensevoice.axera/python/models/SenseVoice/sensevoice_ax650
AX_ASR_LANGUAGE=zh
ENABLE_AX_TTS=true
AX_TTS_MODEL_PATH=/mnt/axera/models/kokoro
AX_TTS_ESPEAK_DATA_PATH=/mnt/axera/models/kokoro-data/espeak-ng-data
AX_TTS_JIEBA_DICT_PATH=/mnt/axera/models/kokoro-data/dict
AX_TTS_MAX_SEQ_LEN=96
AX_TTS_TYPE=KOKORO
AX_TTS_VOICE=zf_xiaoxiao
AX_TTS_LANGUAGE=zh
AX_TTS_SAMPLE_RATE=24000
# 端侧 LLM 默认关闭（云端 LLM）；如需启用设 AXLLM_ENABLE=1
ENABLE_AX_LLM=false
ENABLE_SPEAKER_RECOGNITION=false
ENABLE_WAKE_WORD=false
ENABLE_EMOTION_DETECTION=false
ENABLE_VISION=false
ENABLE_NOISE_REDUCTION=false
EOF
  echo "[deploy] 已生成 Her.axera backend/.env，请填入 DEEPSEEK_API_KEY（或 OPENAI_COMPAT_*）"
fi
# ax_asr / ax_tts wheel（离线安装，板端直连 github 常超时）
python3 -m pip install --no-input --target "$PY_TARGET" \
  "${AXERA_DIR}/wheels/ax650_ax_asr-0.1.0-cp310-cp310-linux_aarch64.whl" \
  "${AXERA_DIR}/wheels/ax_tts-0.1.5-cp310-cp310-linux_aarch64.whl" \
  2>/dev/null || echo "[warn] ax_asr/ax_tts wheel 安装失败，请检查 ${AXERA_DIR}/wheels"

echo "==> 7/8 端侧 LLM（可选）"
if [ "${AXLLM_ENABLE:-0}" = "1" ]; then
  if [ ! -x "${AXERA_DIR}/bin/axllm" ]; then
    curl -fL --retry 3 -o "${AXERA_DIR}/bin/axllm" \
      https://github.com/AXERA-TECH/ax-llm/releases/latest/download/axllm-ax650-linux-arm64
    chmod +x "${AXERA_DIR}/bin/axllm"
  fi
  if [ ! -f "${AXERA_DIR}/bsp/msp_3.6.2/out/lib/libax_sys.so" ]; then
    cd "${AXERA_DIR}/bsp"
    curl -fL --retry 3 -o msp_3.6.2.zip \
      https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/msp_3.6.2.zip
    unzip -qo msp_3.6.2.zip
  fi
  QWEN_DIR="${MODELS_DIR}/Qwen3-0.6B"
  if [ ! -f "${QWEN_DIR}/qwen3_p128_l0_together.axmodel" ]; then
    (cd "${REPO_ROOT}/axera/models" && bash download_models.sh)
  fi
  [ -s "${QWEN_DIR}/config.json" ] || { echo "[deploy] Qwen3-0.6B config.json 缺失"; exit 1; }
else
  echo "[deploy] 默认云端 LLM（Her.axera 网关）；如需端侧 axllm 请设置 AXLLM_ENABLE=1"
fi

echo "==> 8/8 启动全部服务"
bash "${REPO_ROOT}/axera/deploy/start_all.sh"
sleep 8
echo
echo "======== 部署完成 ========"
echo "LLM  : http://<BOARD_IP>:8080/v1/chat/completions （Her.axera 网关，deepseek/openai_compat 云端）"
echo "        （可选端侧 axllm：AXLLM_ENABLE=1 部署，端口 8001）"
echo "ASR  : http://<BOARD_IP>:1000/v1/upload_audio + ws://<BOARD_IP>:1000/v1/ws/vad"
echo "TTS  : http://<BOARD_IP>:5000/ (POST {text,text_language})"
echo "RAG  : http://<BOARD_IP>:8002/ask"
echo "BERT : http://<BOARD_IP>:6007/classify"
echo "把 axera/config.axera.json 中的 <BOARD_IP> 替换后写入 live-2d/config.json 即可使用。"
