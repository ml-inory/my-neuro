#!/usr/bin/env python3
"""AX650 ASR 后端服务（my-neuro axera 分支）。

保持与原 full-hub/asr_api.py 完全一致的 HTTP/WS 契约：
  - POST /v1/upload_audio   上传音频文件 -> {"status","filename","text"}
  - WS   /v1/ws/vad         512 个 float32 采样 -> {"is_speech","probability"}
  - GET  /hotwords         查看热词
  - POST /hotwords/reload   重新加载热词文件
  - GET  /vad/status        VAD 连接状态

推理后端（Her.axera 统一提供，见 axera/deps/Her.axera）：
  - ASR：转发 Her.axera POST /v1/audio/transcriptions（ax_asr，SenseVoice NPU）
  - VAD：本服务内置 Silero-VAD ONNX（CPU，onnxruntime），保持流式契约
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from queue import Queue

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    import onnxruntime as ort
except ImportError:
    ort = None
try:
    import requests
except ImportError:
    requests = None

SAMPLE_RATE = 16000
WINDOW_SIZE = 512
VAD_THRESHOLD = 0.7

app = FastAPI(title="my-neuro AX650 ASR")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 运行时状态
vad_state = {"is_running": False, "active_websockets": set(), "vad": None}
asr_state = {"backend_url": "", "language": "zh"}
hotword_state = {"hotwords": "", "file": ""}


class SileroVAD:
    """Silero-VAD v4/v5 ONNX 封装。

    v5 模型输入: input [1,512] float32 / state [2,1,128] float32 / sr [] int64，
    输出: output [1,1] 语音概率 / stateN（下一帧的 state）。
    v4 模型仅 input [1,512] -> output [1,1]。
    """

    def __init__(self, onnx_path: str):
        if ort is None:
            raise RuntimeError("onnxruntime 未安装，无法运行 Silero-VAD")
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def predict(self, audio: np.ndarray, state: np.ndarray) -> tuple:
        """返回 (语音概率, 下一帧 state)。"""
        feed = {}
        for name in self.inputs:
            if name == "input":
                feed[name] = np.asarray(audio, dtype=np.float32).reshape(1, -1)
            elif name == "sr":
                feed[name] = np.array([SAMPLE_RATE], dtype=np.int64)
            elif name == "state":
                feed[name] = state
        outs = self.sess.run(self.output_names, feed)
        prob = float(np.asarray(outs[0]).reshape(-1)[0])
        new_state = outs[1] if len(outs) > 1 else state
        return prob, new_state


def load_hotwords(path: str) -> str:
    """读取 hotwords.txt（每行 '词语 权重' 或纯词语），返回空格分隔的词语串。"""
    if not os.path.exists(path):
        print(f"[asr] 热词文件不存在: {path}")
        return ""
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 格式：词语 权重（权重只用于 funasr，SenseVoice 只取词）
            words.append(line.split()[0])
    print(f"[asr] 已加载 {len(words)} 个热词")
    return " ".join(words)


def clean_sensevoice_text(text: str) -> str:
    """去掉 SenseVoice 输出的语言/情感/事件特殊标签（如 <|zh|> <|HAPPY|>）。"""
    text = re.sub(r"<\|[^|]*\|>", "", text)
    return text.strip()


@app.on_event("startup")
async def startup():
    # 通过环境变量注入部署路径
    repo = Path(os.environ.get("AXERA_REPO", Path(__file__).resolve().parent.parent))
    vad_model = os.environ.get(
        "AXERA_VAD_MODEL",
        str(repo / "axera" / "models" / "vad" / "silero_vad.onnx"),
    )
    backend_url = os.environ.get("AXERA_HER_BACKEND", "http://127.0.0.1:8080/v1")
    hotwords_file = os.environ.get(
        "AXERA_HOTWORDS_FILE",
        str(repo / "full-hub" / "hotwords.txt"),
    )
    language = os.environ.get("AXERA_ASR_LANGUAGE", "zh")

    hotword_state["file"] = hotwords_file
    hotword_state["hotwords"] = load_hotwords(hotwords_file)
    vad_state["vad"] = SileroVAD(vad_model)
    asr_state["language"] = language
    asr_state["backend_url"] = backend_url
    print(f"[asr] VAD 就绪；ASR 转发 Her.axera {backend_url}/audio/transcriptions")


@app.websocket("/v1/ws/vad")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    vad_state["active_websockets"].add(websocket)
    ws_state = np.zeros((2, 1, 128), dtype=np.float32)
    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                audio = np.frombuffer(data, dtype=np.float32).copy()
                if len(audio) == WINDOW_SIZE:
                    prob, ws_state = vad_state["vad"].predict(audio, ws_state)
                    await websocket.send_text(
                        json.dumps(
                            {
                                "is_speech": prob > VAD_THRESHOLD,
                                "probability": float(prob),
                            }
                        )
                    )
            except WebSocketDisconnect:
                break
    finally:
        vad_state["active_websockets"].discard(websocket)


@app.post("/v1/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if requests is None:
            return {"status": "error", "message": "requests 未安装"}
        # 转发 Her.axera /v1/audio/transcriptions（ax_asr，SenseVoice NPU）
        resp = requests.post(
            f"{asr_state['backend_url']}/audio/transcriptions",
            files={"file": (file.filename or "audio.wav", audio_bytes)},
            data={"model": "ax_asr_sensevoice", "language": asr_state["language"]},
            timeout=120,
        )
        resp.raise_for_status()
        text = (resp.json() or {}).get("text", "").strip()
        if not text:
            return {"status": "error", "filename": file.filename or "uploaded_audio", "message": "语音识别失败"}
        return {"status": "success", "filename": file.filename or "uploaded_audio", "text": text}
    except Exception as e:
        print(f"[asr] 处理音频出错: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/hotwords")
def get_hotwords():
    return {"hotwords": hotword_state["hotwords"], "file": hotword_state["file"]}


@app.post("/hotwords/reload")
def reload_hotwords():
    hotword_state["hotwords"] = load_hotwords(hotword_state["file"])
    return {"status": "success", "hotwords": hotword_state["hotwords"]}


@app.get("/vad/status")
def get_status():
    vad_state["active_websockets"] = {
        ws for ws in vad_state["active_websockets"] if ws.client_state.state.name != "DISCONNECTED"
    }
    return {
        "is_running": bool(vad_state["active_websockets"]),
        "active_connections": len(vad_state["active_websockets"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
