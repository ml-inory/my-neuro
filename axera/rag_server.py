#!/usr/bin/env python3
"""AX650 RAG 后端服务（my-neuro axera 分支）。

保持与原 full-hub/run_rag.py 一致的契约（端口 8002）：
  GET  /         服务状态
  POST /encode   文本 -> embedding（bge-m3 dense）
  POST /similarity 两段文本相似度
  POST /ask      知识库检索（记忆库.txt）
  GET  /health   健康检查

推理后端：BAAI/bge-m3 AXMODEL（AXERA-TECH/bge-m3，NPU，w8a16）。
"""
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import axengine as axe
except ImportError:
    axe = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

app = FastAPI(title="my-neuro AX650 RAG (bge-m3)")

MAX_LENGTH = 512


class BGEM3Ax:
    """bge-m3 axmodel 封装：dense 向量（L2 归一化后做余弦相似度）。"""

    def __init__(self, axmodel: str, tokenizer_name: str):
        if axe is None:
            raise RuntimeError("axengine 未安装")
        if AutoTokenizer is None:
            raise RuntimeError("transformers 未安装（tokenizer 需要）")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.session = axe.InferenceSession(axmodel, providers=["AxEngineExecutionProvider"])

    def encode_one(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            [text],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int32)
        dense_vecs, _, _ = self.session.run(None, {"input_ids": input_ids})
        vec = dense_vecs[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.encode_one(t) for t in texts])


def load_knowledge_base(file_path: str) -> List[str]:
    """与原 run_rag.py 一致：10 个以上连续横线分段。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        sections = re.split(r"\s*-{10,}\s*", content)
        paragraphs = [s.strip() for s in sections if s.strip() and len(s.strip()) > 10]
        print(f"[rag] 知识库加载完成，共 {len(paragraphs)} 个段落")
        return paragraphs
    except Exception as e:
        print(f"[rag] 加载知识库失败: {e}")
        return []


rag_state = {
    "model": None,
    "knowledge_base": [],
    "knowledge_embeddings": None,
    "kb_file": "",
    "lock": threading.Lock(),
}


def reload_knowledge_base():
    with rag_state["lock"]:
        paragraphs = load_knowledge_base(rag_state["kb_file"])
        if paragraphs and rag_state["model"] is not None:
            rag_state["knowledge_base"] = paragraphs
            rag_state["knowledge_embeddings"] = rag_state["model"].encode(paragraphs)
            print(f"[rag] 知识库嵌入完成: {len(paragraphs)} 段")


@app.on_event("startup")
async def startup():
    repo = Path(os.environ.get("AXERA_REPO", Path(__file__).resolve().parent.parent))
    axmodel = os.environ.get(
        "AXERA_BGE_AXMODEL",
        str(repo / "axera" / "models" / "bge-m3" / "model" / "bge-m3_u16_npu3.axmodel"),
    )
    tokenizer = os.environ.get("AXERA_BGE_TOKENIZER", "BAAI/bge-m3")
    kb_file = os.environ.get("AXERA_KB_FILE", str(repo / "live-2d" / "AI记录室" / "记忆库.txt"))
    rag_state["kb_file"] = kb_file
    rag_state["model"] = BGEM3Ax(str(axmodel), tokenizer)
    reload_knowledge_base()

    # 文件变化轮询（watchdog 不可用时兜底）
    def _poll():
        last_mtime = -1
        while True:
            try:
                mtime = os.path.getmtime(kb_file)
                if mtime != last_mtime:
                    last_mtime = mtime
                    reload_knowledge_base()
            except Exception:
                pass
            time.sleep(5)

    threading.Thread(target=_poll, daemon=True).start()


class TextRequest(BaseModel):
    text: str


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int
    processing_time: float


class AnswerResponse(BaseModel):
    question: str
    relevant_passages: List[Dict[str, Any]]
    processing_time: float


class SimilarityResponse(BaseModel):
    similarity: float
    processing_time: float


@app.get("/")
async def root():
    return {
        "message": "BGE API服务运行中",
        "model_loaded": rag_state["model"] is not None,
        "knowledge_base_size": len(rag_state["knowledge_base"]),
    }


@app.post("/encode", response_model=EmbeddingResponse)
async def encode_text(request: TextRequest):
    if rag_state["model"] is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    start = time.time()
    embedding = rag_state["model"].encode_one(request.text)
    return EmbeddingResponse(
        embedding=embedding.tolist(),
        dimension=len(embedding),
        processing_time=time.time() - start,
    )


@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    if rag_state["model"] is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    start = time.time()
    a, b = rag_state["model"].encode([request.text1, request.text2])
    similarity = float(np.dot(a, b))
    return SimilarityResponse(similarity=similarity, processing_time=time.time() - start)


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if rag_state["model"] is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    if not rag_state["knowledge_base"]:
        raise HTTPException(status_code=404, detail="知识库未加载")
    with rag_state["lock"]:
        start = time.time()
        q = rag_state["model"].encode_one(request.question)
        scores = rag_state["knowledge_embeddings"] @ q
        top_indices = np.argsort(scores)[::-1][: request.top_k]
        relevant_passages = [
            {
                "rank": i + 1,
                "similarity": float(scores[idx]),
                "content": rag_state["knowledge_base"][idx],
            }
            for i, idx in enumerate(top_indices)
        ]
        return AnswerResponse(
            question=request.question,
            relevant_passages=relevant_passages,
            processing_time=time.time() - start,
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": rag_state["model"] is not None,
        "knowledge_base_loaded": len(rag_state["knowledge_base"]) > 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
