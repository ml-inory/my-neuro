#!/usr/bin/env bash
# 用 Pulsar2 编译 Omni_fn_bert ONNX -> AX650 AXMODEL。
# 用法（x86 开发机，需 docker + pulsar2:7.0 镜像）：
#   bash axera/models/compile_axmodel.sh [bert模型目录] [输出目录]
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
IMAGE="${PULSAR2_IMAGE:-pulsar2:7.0}"
MODEL_DIR="${1:-${ROOT}/axera/models/bert}"
OUT_DIR="${2:-${ROOT}/axera/models/bert/compile}"

mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/build.json" <<EOF
{
  "input": "/workspace/bert/model.onnx",
  "output_dir": "/workspace/compile",
  "output_name": "omni_fn_bert.axmodel",
  "model_type": "ONNX",
  "target_hardware": "AX650",
  "npu_mode": "NPU3",
  "quant": {
    "calibration_method": "MinMax",
    "transformer_opt_level": 1,
    "input_configs": [
      {"tensor_name": "input_ids",       "calibration_dataset": "/workspace/bert/calib/input_ids.tar.gz",       "calibration_size": 16, "calibration_format": "Numpy"},
      {"tensor_name": "attention_mask",  "calibration_dataset": "/workspace/bert/calib/attention_mask.tar.gz",  "calibration_size": 16, "calibration_format": "Numpy"},
      {"tensor_name": "token_type_ids",  "calibration_dataset": "/workspace/bert/calib/token_type_ids.tar.gz",  "calibration_size": 16, "calibration_format": "Numpy"},
      {"tensor_name": "position_ids",    "calibration_dataset": "/workspace/bert/calib/position_ids.tar.gz",    "calibration_size": 16, "calibration_format": "Numpy"},
      {"tensor_name": "task_type_id",    "calibration_dataset": "/workspace/bert/calib/task_type_id.tar.gz",    "calibration_size": 16, "calibration_format": "Numpy"}
    ],
    "highest_mix_precision": false
  },
  "input_processors": [],
  "output_processors": [],
  "compiler": {"check": 3}
}
EOF

echo "[compile] 编译中（pulsar2 build）..."
docker run --rm -v "$MODEL_DIR":/workspace/bert -v "$OUT_DIR":/workspace/compile "$IMAGE" \
  pulsar2 build --config /workspace/compile/build.json 2>&1 | tee "$OUT_DIR/compile.log" | tail -25

if [ -f "$OUT_DIR/omni_fn_bert.axmodel" ]; then
  echo "[compile] OK: $OUT_DIR/omni_fn_bert.axmodel"
  du -h "$OUT_DIR/omni_fn_bert.axmodel"
else
  echo "[compile] 失败，见 $OUT_DIR/compile.log" >&2
  exit 1
fi
