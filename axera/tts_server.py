#!/usr/bin/env python3
"""AX650 TTS 后端服务（my-neuro axera 分支）。

保持与原 GPT-SoVITS api.py 一致的前端调用契约：
  POST /     （以及 POST /tts） JSON {"text": "...", "text_language": "zh"} -> audio/wav

推理后端：MeloTTS（ml-inory/melotts.axera）
  - encoder：ONNX（CPU）
  - decoder：AXMODEL（NPU，axengine）
  - bert：可选，bert-hidden-u16-zh.axmodel（NPU，若 encoder 需要 bert 输入）
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel

app = FastAPI(title="my-neuro AX650 TTS")

LANG_MAP = {
    "zh": "ZH",
    "zh-cn": "ZH",
    "chinese": "ZH",
    "en": "EN",
    "english": "EN",
    "ja": "JP",
    "jp": "JP",
    "japanese": "JP",
    "ko": "KR",
    "kr": "KR",
    "korean": "KR",
    "yue": "YUE",
    "auto": "ZH",
}

tts_state = {"tts": None, "sample_rate": 44100, "speed": 1.0}


class TTSRequest(BaseModel):
    text: str = ""
    text_language: str = "zh"
    text_lang: str = ""
    sentence: str = ""
    language: str = ""
    speed: float = 1.0
    sample_rate: int = 44100


def resolve_language(req: TTSRequest) -> str:
    raw = (req.text_language or req.language or req.text_lang or "zh").lower().strip()
    return LANG_MAP.get(raw, "ZH")


def load_melotts(encoder: str, decoder: str, bert: str, bert_tokenizer: str, dec_len: int):
    melotts_py = Path(__file__).resolve().parent / "deps" / "melotts.axera" / "python"
    if not melotts_py.exists():
        raise RuntimeError(f"找不到 {melotts_py}，请先运行 axera/deploy/install_board.sh 拉取依赖")
    # melotts.py 内部用相对路径读取 ../models/g-*.bin
    os.chdir(melotts_py)
    sys.path.insert(0, str(melotts_py))
    from melotts import MeloTTS  # noqa: E402

    return MeloTTS(
        encoder,
        decoder,
        "ZH",
        dec_len,
        bert_model=bert,
        bert_model_id=bert_tokenizer,
    )


@app.on_event("startup")
async def startup():
    repo = Path(os.environ.get("AXERA_REPO", Path(__file__).resolve().parent.parent))
    melotts_dir = os.environ.get("AXERA_MELOTTS_DIR", str(repo / "axera" / "deps" / "melotts.axera" / "models"))
    encoder = os.environ.get("AXERA_MELOTTS_ENCODER", os.path.join(melotts_dir, "encoder-zh.onnx"))
    decoder = os.environ.get("AXERA_MELOTTS_DECODER", os.path.join(melotts_dir, "decoder-zh.axmodel"))
    bert = os.environ.get("AXERA_MELOTTS_BERT") or None
    bert_tokenizer = os.environ.get("AXERA_MELOTTS_BERT_TOKENIZER", "hfl/chinese-roberta-wwm-ext-large")
    dec_len = int(os.environ.get("AXERA_MELOTTS_DEC_LEN", "128"))
    tts_state["sample_rate"] = int(os.environ.get("AXERA_MELOTTS_SAMPLE_RATE", "44100"))
    tts_state["speed"] = float(os.environ.get("AXERA_MELOTTS_SPEED", "1.0"))
    tts_state["tts"] = load_melotts(encoder, decoder, bert, bert_tokenizer, dec_len)


@app.post("/")
@app.post("/tts")
async def tts(req: TTSRequest):
    text = (req.text or req.sentence or "").strip()
    if not text:
        return Response(status_code=400, content="field 'text' is required")
    try:
        audio = tts_state["tts"].run(
            text,
            speed=req.speed if req.speed else tts_state["speed"],
            sample_rate=req.sample_rate if req.sample_rate else tts_state["sample_rate"],
        )
        audio = np.asarray(audio, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, tts_state["sample_rate"], format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")
    except Exception as e:
        print(f"[tts] 合成失败: {e}")
        return Response(status_code=500, content=f"TTS failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_state["tts"] is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
