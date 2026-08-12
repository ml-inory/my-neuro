#!/usr/bin/env python3
"""AX650 BERT 后端服务（my-neuro axera 分支）。

保持与原 full-hub/omni_bert_api.py 一致的契约（端口 6007）：
  POST /classify {"text": "..."} -> {"text", "Vision", "core memory"}

模型：morelle/Omni_fn_bert（Ernie-3.0-base-zh 多标签二分类）。
推理：优先 AXMODEL（NPU，axengine）；未提供 axmodel 时回退 ONNX（CPU，onnxruntime）。
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

try:
    import axengine as axe
except ImportError:
    axe = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

app = FastAPI(title="my-neuro AX650 BERT")

LABEL_MAPPING = {"0": "否", "1": "是"}
LABELS = ["Vision", "core memory"]
MAX_LENGTH = 512


class Classifier:
    """Ernie-3.0 分类器封装，自动适配 NPU axmodel 或 ONNX。"""

    def __init__(self, tokenizer_dir: str, axmodel: Optional[str], onnx_model: Optional[str]):
        if AutoTokenizer is None:
            raise RuntimeError("transformers 未安装")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.session = None
        self.backend = None
        if axmodel and os.path.exists(axmodel):
            if axe is None:
                raise RuntimeError("axengine 未安装，无法运行 AXMODEL")
            self.session = axe.InferenceSession(axmodel, providers=["AxEngineExecutionProvider"])
            self.backend = "npu"
        elif onnx_model and os.path.exists(onnx_model):
            if ort is None:
                raise RuntimeError("onnxruntime 未安装，无法回退 CPU")
            self.session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
            self.backend = "cpu"
        else:
            raise RuntimeError(
                "未找到 BERT 模型。请设置 AXERA_BERT_AXMODEL（NPU）或 AXERA_BERT_ONNX（CPU 回退）。"
            )
        self.input_names = {i.name for i in self.session.get_inputs()}
        print(f"[bert] 后端: {self.backend}，输入: {sorted(self.input_names)}")

    def _build_inputs(self, text: str):
        encoded = self.tokenizer(
            [text],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="np",
        )
        n = encoded["input_ids"].shape[1]
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int32),
            "token_type_ids": encoded["token_type_ids"].astype(np.int32),
            "attention_mask": encoded["attention_mask"].astype(np.int32),
            "task_type_id": np.zeros((1,), dtype=np.int32),
            "position_ids": np.arange(n, dtype=np.int32)[None, :],
        }
        return {k: v for k, v in inputs.items() if k in self.input_names}

    def classify(self, text: str) -> dict:
        feed = self._build_inputs(text)
        logits = self.session.run(None, feed)[0]
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        probs = probs.reshape(-1)
        preds = (probs > 0.5).astype(int)
        result = {"text": text}
        for i, label in enumerate(LABELS):
            result[label] = LABEL_MAPPING[str(preds[i] if i < len(preds) else 0)]
        return result


bert_state = {"model": None}


@app.on_event("startup")
async def startup():
    repo = Path(os.environ.get("AXERA_REPO", Path(__file__).resolve().parent.parent))
    bert_dir = os.environ.get("AXERA_BERT_DIR", str(repo / "axera" / "models" / "bert"))
    axmodel = os.environ.get("AXERA_BERT_AXMODEL", os.path.join(bert_dir, "omni_fn_bert.axmodel"))
    onnx_model = os.environ.get("AXERA_BERT_ONNX", os.path.join(bert_dir, "omni_fn_bert.onnx"))
    bert_state["model"] = Classifier(bert_dir, axmodel, onnx_model)


class TextInput(BaseModel):
    text: str


@app.post("/classify")
async def classify_emotion(input_data: TextInput):
    return bert_state["model"].classify(input_data.text)


@app.get("/health")
async def health():
    return {"status": "healthy", "backend": bert_state["model"].backend if bert_state["model"] else None}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6007)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
