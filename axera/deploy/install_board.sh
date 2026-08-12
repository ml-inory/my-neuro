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
mkdir -p "$PY_TARGET"
python3 -m pip install --no-input --target "$PY_TARGET" --upgrade pip setuptools wheel 2>/dev/null || true
python3 -m pip install --no-input --target "$PY_TARGET" -r "${REPO_ROOT}/axera/requirements.txt"
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

echo "==> 6/8 安装 axllm 预编译二进制"
if [ ! -x "${AXERA_DIR}/bin/axllm" ]; then
  curl -fL --retry 3 -o "${AXERA_DIR}/bin/axllm" \
    https://github.com/AXERA-TECH/ax-llm/releases/latest/download/axllm-ax650-linux-arm64
  chmod +x "${AXERA_DIR}/bin/axllm"
fi
# BSP 运行库（axllm 动态依赖）
if [ ! -f "${AXERA_DIR}/bsp/msp_3.6.2/out/lib/libax_sys.so" ]; then
  cd "${AXERA_DIR}/bsp"
  curl -fL --retry 3 -o msp_3.6.2.zip \
    https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/msp_3.6.2.zip
  unzip -qo msp_3.6.2.zip
fi

echo "==> 7/8 准备 Qwen2.5-1.5B 模型目录（若未下载）"
QWEN_DIR="${MODELS_DIR}/Qwen2.5-1.5B-Instruct"
QWEN_CFG="${QWEN_DIR}/config.json"
if [ ! -f "${QWEN_DIR}/qwen2.5-1.5b-ctx-int4-ax650/qwen2_p128_l0_together.axmodel" ]; then
  (cd "${REPO_ROOT}/axera/models" && bash download_models.sh)
fi
# axllm 需要 config.json（该 HF/ModelScope 仓库的 config.json 为空，这里生成）
if [ ! -s "${QWEN_CFG}" ]; then
  cat > "${QWEN_CFG}" <<EOF
{
  "model_name": "Qwen2.5-1.5B-Instruct",
  "tokenizer_type": "Qwen2_5",
  "url_tokenizer_model": "qwen2.5_tokenizer/tokenizer.json",
  "template_filename_axmodel": "qwen2.5-1.5b-ctx-int4-ax650/qwen2_p128_l%d_together.axmodel",
  "axmodel_num": 28,
  "filename_post_axmodel": "qwen2.5-1.5b-ctx-int4-ax650/qwen2_post.axmodel",
  "filename_tokens_embed": "qwen2.5-1.5b-ctx-int4-ax650/model.embed_tokens.weight.bfloat16.bin",
  "tokens_embed_num": 151936,
  "tokens_embed_size": 1536
}
EOF
  echo "[deploy] 已生成 axllm config.json (w4a16)"
fi

echo "==> 8/8 启动全部服务"
bash "${REPO_ROOT}/axera/deploy/start_all.sh"
sleep 8
echo
echo "======== 部署完成 ========"
echo "LLM  : http://<BOARD_IP>:8000/v1/models （axllm，Qwen2.5-1.5B）"
echo "ASR  : http://<BOARD_IP>:1000/v1/upload_audio + ws://<BOARD_IP>:1000/v1/ws/vad"
echo "TTS  : http://<BOARD_IP>:5000/ (POST {text,text_language})"
echo "RAG  : http://<BOARD_IP>:8002/ask"
echo "BERT : http://<BOARD_IP>:6007/classify"
echo "把 axera/config.axera.json 中的 <BOARD_IP> 替换后写入 live-2d/config.json 即可使用。"
