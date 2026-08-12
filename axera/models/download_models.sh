#!/usr/bin/env bash
# 下载 my-neuro axera 后端所需的全部模型到 axera/models/（板端与 x86 通用）。
#
# 模型清单:
#   VAD        morelle/my-neuro-vad      silero_vad.onnx             (~2MB,   ModelScope)
#   BERT       morelle/Omni_fn_bert      Ernie-3.0-base 分类器        (~450MB, ModelScope)
#   RAG        AXERA-TECH/bge-m3         bge-m3_u16_npu3.axmodel     (~850MB, HuggingFace)
#   LLM        AXERA-TECH/Qwen3-0.6B  axllm 模型目录                (~1GB,   ModelScope 优先)
#   ASR        AXERA-TECH/SenseVoice     sensevoice_ax650            (由 sensevoice.axera 脚本下载)
#   TTS        ml-inory/melotts.axera    encoder-onnx + decoder-axmodel (由 melotts.axera 脚本下载)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${AXERA_MODELS_DIR:-$SCRIPT_DIR}"
HFD="${ROOT}/axera/scripts/download_hf.sh"
cd "$MODELS_DIR"
mkdir -p vad bert bge-m3 Qwen3-0.6B

# 1) Silero VAD（ModelScope 单文件）
if [ ! -f vad/silero_vad.onnx ]; then
  echo "[models] 下载 Silero VAD..."
  python3 - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
import os, shutil
d = snapshot_download('morelle/my-neuro-vad', allow_patterns=['snakers4_silero-vad_master/src/silero_vad/data/silero_vad.onnx'])
src = os.path.join(d, 'snakers4_silero-vad_master/src/silero_vad/data/silero_vad.onnx')
shutil.copy(src, 'vad/silero_vad.onnx')
print('VAD -> vad/silero_vad.onnx')
PY
fi

# 2) Omni_fn_bert（ModelScope，权重 + tokenizer）
if [ ! -f bert/config.json ] || [ ! -f bert/omni_fn_bert.axmodel ]; then
  echo "[models] 下载 Omni_fn_bert..."
  python3 - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
import shutil, os
d = snapshot_download('morelle/Omni_fn_bert')
for f in os.listdir(d):
    shutil.copy(os.path.join(d, f), os.path.join('bert', f))
print('BERT -> bert/')
PY
fi

# 3) bge-m3 axmodel（AX650 NPU）
if [ ! -f bge-m3/model/bge-m3_u16_npu3.axmodel ]; then
  echo "[models] 下载 bge-m3 axmodel (AX650)..."
  bash "$HFD" AXERA-TECH/bge-m3 --local-dir bge-m3 -x 8
fi

# 4) Qwen3-0.6B ax-llm 模型目录（AX650 官方转换，28 层 w8a16，~1GB）
QWEN_DIR="Qwen3-0.6B"
QWEN_KEY="${QWEN_DIR}/qwen3_p128_l0_together.axmodel"
if [ ! -f "$QWEN_KEY" ]; then
  echo "[models] 下载 Qwen3-0.6B (AX650, axllm)..."
  python3 - <<'PY'
import os
try:
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(
        'AXERA-TECH/Qwen3-0.6B',
        local_dir='Qwen3-0.6B',
        allow_patterns=[
            'qwen3_p128_l*_together.axmodel',
            'qwen3_post.axmodel',
            'model.embed_tokens.weight.bfloat16.bin',
            'qwen3_tokenizer.txt',
            'config.json', 'post_config.json', 'configuration.json',
        ],
    )
    print('Qwen3 (ModelScope) -> Qwen3-0.6B/')
except Exception as e:
    print(f'ModelScope 下载失败，回退 HuggingFace: {e}')
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'huggingface_hub.commands.huggingface_cli',
        'download', 'AXERA-TECH/Qwen3-0.6B',
        '--local-dir', 'Qwen3-0.6B'])
PY
fi

# 5) SenseVoice（由 sensevoice.axera 自带脚本下载，ASR 服务会引用）
if [ -d "${ROOT}/axera/deps/sensevoice.axera" ] && [ ! -f "${ROOT}/axera/deps/sensevoice.axera/python/models/SenseVoice/sensevoice_ax650/sensevoice.axmodel" ]; then
  (cd "${ROOT}/axera/deps/sensevoice.axera" && bash download_models.sh)
fi

# 6) MeloTTS（由 melotts.axera 自带脚本下载，TTS 服务会引用）
if [ -d "${ROOT}/axera/deps/melotts.axera" ] && [ ! -f "${ROOT}/axera/deps/melotts.axera/models/decoder-zh.axmodel" ]; then
  (cd "${ROOT}/axera/deps/melotts.axera" && bash download_models.sh)
fi

echo "[models] 全部模型就绪："
du -sh vad bert bge-m3 Qwen2.5-1.5B-Instruct 2>/dev/null
