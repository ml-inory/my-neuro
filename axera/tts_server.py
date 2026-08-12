#!/usr/bin/env python3
"""AX650 TTS 后端服务（my-neuro axera 分支）。

保持与原 GPT-SoVITS api.py 一致的前端调用契约：
  POST /     （以及 POST /tts） JSON {"text": "...", "text_language": "zh"} -> audio/wav

推理后端：转发 Her.axera POST /v1/audio/speech（ax_tts，Kokoro NPU；
未启用 ax_tts 时回退 edge_tts 云端）。
"""
import argparse
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel

try:
    import requests
except ImportError:
    requests = None

app = FastAPI(title="my-neuro AX650 TTS")

tts_state = {"backend_url": "", "voice": "female_default", "model": "ax_tts_kokoro"}


class TTSRequest(BaseModel):
    text: str = ""
    text_language: str = "zh"
    text_lang: str = ""
    sentence: str = ""
    language: str = ""


@app.on_event("startup")
async def startup():
    repo = Path(os.environ.get("AXERA_REPO", Path(__file__).resolve().parent.parent))
    tts_state["backend_url"] = os.environ.get("AXERA_HER_BACKEND", "http://127.0.0.1:8080/v1")
    tts_state["voice"] = os.environ.get("AXERA_TTS_VOICE", "female_default")
    tts_state["model"] = os.environ.get("AXERA_TTS_MODEL", "ax_tts_kokoro")
    print(f"[tts] TTS 转发 Her.axera {tts_state['backend_url']}/audio/speech (voice={tts_state['voice']})")


@app.post("/")
@app.post("/tts")
async def tts(req: TTSRequest):
    text = (req.text or req.sentence or "").strip()
    if not text:
        return Response(status_code=400, content="field 'text' is required")
    try:
        if requests is None:
            return Response(status_code=500, content="requests 未安装")
        resp = requests.post(
            f"{tts_state['backend_url']}/audio/speech",
            json={
                "model": tts_state["model"],
                "input": text,
                "voice": tts_state["voice"],
                "language": (req.text_language or req.language or "zh"),
                "response_format": "wav",
            },
            timeout=120,
        )
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "audio/wav")
        return Response(content=resp.content, media_type=ctype)
    except Exception as e:
        print(f"[tts] 合成失败: {e}")
        return Response(status_code=500, content=f"TTS failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "backend": tts_state["backend_url"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
