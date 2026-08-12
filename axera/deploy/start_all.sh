#!/usr/bin/env bash
# 启动 my-neuro axera 全部后端服务（nohup + pid 文件，不依赖 systemd）
set -euo pipefail

cd "$(dirname "$0")/.."
AXERA_DIR="${AXERA_DIR:-/mnt/axera}"
PY_TARGET="${AXERA_DIR}/pylib/site-packages"
LOG_DIR="${AXERA_DIR}/logs"
REPO_ROOT="$(cd .. && pwd)"

[ -d "$PY_TARGET" ] || { echo "未找到 $PY_TARGET，请先运行 install_board.sh"; exit 1; }
mkdir -p "$LOG_DIR"

export AXERA_REPO="$REPO_ROOT"
export PYTHONPATH="$PY_TARGET:${PYTHONPATH:-}"
export AXERA_VAD_MODEL="${AXERA_DIR}/models/vad/silero_vad.onnx"
export AXERA_BERT_DIR="${AXERA_DIR}/models/bert"
export AXERA_BERT_AXMODEL="${AXERA_DIR}/models/bert/omni_fn_bert.axmodel"
export AXERA_BERT_ONNX="${AXERA_DIR}/models/bert/model.onnx"
export AXERA_BGE_AXMODEL="${AXERA_DIR}/models/bge-m3/model/bge-m3_u16_npu3.axmodel"
export AXERA_BGE_TOKENIZER="${AXERA_DIR}/models/bge-m3-tokenizer"
export AXERA_KB_FILE="${REPO_ROOT}/AI记录室/记忆库.txt"
export AXERA_HER_BACKEND="${AXERA_HER_BACKEND:-http://127.0.0.1:8080/v1}"

start_one() {
  local name="$1"; shift
  local pidfile="${AXERA_DIR}/logs/${name}.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[skip] $name 已在运行"
    return
  fi
  echo "[start] $name"
  nohup "$@" >>"${LOG_DIR}/${name}.log" 2>&1 &
  echo $! > "$pidfile"
}

start_one asr python3 "$REPO_ROOT/axera/asr_server.py"
start_one tts python3 "$REPO_ROOT/axera/tts_server.py"
start_one rag python3 "$REPO_ROOT/axera/rag_server.py"
start_one bert python3 "$REPO_ROOT/axera/bert_server.py"

# Her.axera 云端 LLM 网关（OpenAI 兼容 /v1/chat/completions，deepseek / openai_compat）
HER_DIR="${AXERA_DIR}/deps/Her.axera"
if [ -d "${HER_DIR}/backend" ]; then
  start_one her-axera python3 -m uvicorn app.main:app \
    --app-dir "${HER_DIR}/backend" --host 0.0.0.0 --port "${HER_AXERA_PORT:-8080}"
else
  echo "[warn] Her.axera 未部署，跳过云端 LLM 网关（请运行 install_board.sh）"
fi

# axllm（端侧 LLM，可选：AXLLM_ENABLE=1 部署后自动启用）
AXLLM="${AXERA_DIR}/bin/axllm"
if [ "${AXLLM_ENABLE:-0}" = "1" ] && [ -x "$AXLLM" ]; then
  export LD_LIBRARY_PATH="${AXERA_DIR}/bsp/msp_3.6.2/out/lib:/soc/lib:${LD_LIBRARY_PATH:-}"
  start_one axllm "$AXLLM" serve "${AXERA_DIR}/models/Qwen3-0.6B" --port "${AXLLM_PORT:-8001}"
fi

echo "服务 PID: $(cat "${AXERA_DIR}"/logs/*.pid 2>/dev/null | tr '\n' ' ')"
