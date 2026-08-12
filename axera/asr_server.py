#!/usr/bin/env python3
"""AX650 ASR 后端服务（my-neuro axera 分支）。

保持与原 full-hub/asr_api.py 完全一致的 HTTP/WS 契约：
  - POST /v1/upload_audio   上传音频文件 -> {"status","filename","text"}
  - WS   /v1/ws/vad         512 个 float32 采样 -> {"is_speech","probability"}
  - GET  /hotwords         查看热词
  - POST /hotwords/reload   重新加载热词文件
  - GET  /vad/status        VAD 连接状态

模型推理全部落在 AX650 上：
  - ASR：SenseVoice-Small AXMODEL（NPU，axengine），支持中/英/粤/日/韩 + 自动标点
  - VAD：Silero-VAD ONNX（CPU，onnxruntime）

依赖代码来自 ml-inory/sensevoice.axera（MIT），部署脚本会自动 clone 到
axera/deps/sensevoice.axera。
"""
import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from queue import Queue

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    import onnxruntime as ort
except ImportError:
    ort = None

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
asr_state = {"model": None, "language": "zh"}
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


def load_sensevoice(model_dir: str, language: str, hotwords: str):
    """加载 SenseVoice AXMODEL（NPU）。"""
    model_root = Path(model_dir)
    model_path = model_root / "sensevoice.axmodel"
    assert model_path.exists(), f"找不到 {model_path}，请先运行 axera/models/download_models.sh"
    cmvn = model_root / "am.mvn"
    bpe = model_root / "chn_jpn_yue_eng_ko_spectok.bpe.model"
    tokens = model_root / "tokens.txt"

    sys.path.insert(0, str(Path(__file__).resolve().parent / "deps" / "sensevoice.axera" / "python"))
    from SenseVoiceAx import SenseVoiceAx  # noqa: E402

    hot = hotwords.split() if hotwords else None
    if hot:
        try:
            import asr_decoder  # noqa: F401
            import online_fbank  # noqa: F401
        except ImportError:
            print("[asr] 未安装 asr_decoder/online-fbank（源码包需编译），本板禁用热词")
            hot = None
    model = SenseVoiceAx(
        str(model_path),
        str(cmvn),
        str(tokens),
        str(bpe),
        max_seq_len=256,
        beam_size=3,
        hot_words=hot,
        streaming=False,
    )
    print(f"[asr] SenseVoice 加载完成 (chip=ax650, language={language})")
    return model


@app.on_event("startup")
async def startup():
    # 通过环境变量注入部署路径
    repo = Path(os.environ.get("AXERA_REPO", Path(__file__).resolve().parent.parent))
    model_dir = os.environ.get(
        "AXERA_SENSEVOICE_DIR",
        str(repo / "axera" / "deps" / "sensevoice.axera" / "python" / "models" / "SenseVoice" / "sensevoice_ax650"),
    )
    vad_model = os.environ.get(
        "AXERA_VAD_MODEL",
        str(repo / "axera" / "models" / "vad" / "silero_vad.onnx"),
    )
    hotwords_file = os.environ.get(
        "AXERA_HOTWORDS_FILE",
        str(repo / "full-hub" / "hotwords.txt"),
    )
    language = os.environ.get("AXERA_ASR_LANGUAGE", "zh")

    hotword_state["file"] = hotwords_file
    hotword_state["hotwords"] = load_hotwords(hotwords_file)
    vad_state["vad"] = SileroVAD(vad_model)
    asr_state["language"] = language
    asr_state["model"] = load_sensevoice(model_dir, language, hotword_state["hotwords"])


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
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            audio_data = audio_data.astype(np.float32)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
        except Exception:
            import librosa

            audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            audio_data = audio_data.astype(np.float32)

        if sample_rate != SAMPLE_RATE:
            import librosa

            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

        text = asr_state["model"].infer((audio_data, SAMPLE_RATE), language=asr_state["language"])
        text = clean_sensevoice_text(text)
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
