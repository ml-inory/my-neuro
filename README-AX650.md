# my-neuro AX650 后端移植（axera 分支）

本分支把 my-neuro 的本地 AI 后端搬到 AX650 上运行。ASR / TTS / LLM 统一由
[ml-inory/Her.axera](https://github.com/ml-inory/Her.axera) 提供（ax_asr / ax_tts 走
AX650 NPU；LLM 走云端 deepseek / openai_compat，端侧 axllm 可选），RAG / BERT 由本分支
的适配服务提供（NPU）。前端 Live2D 通过原有 HTTP/WS 接口无缝切换，无需修改前端代码。

## 后端服务与模型清单

| 服务 | 端口 | 原接口（保持不变） | AX650 推理后端 | 模型来源 |
|------|------|--------------------|----------------|----------|
| LLM | 8080 | OpenAI `/v1/chat/completions`（Her.axera 网关） | 云端：deepseek / openai_compat（可选端侧 axllm+Qwen3） | `ml-inory/Her.axera` |
| ASR | 1000 | `/v1/upload_audio`、WS `/v1/ws/vad` | Her.axera `ax_asr`（SenseVoice NPU）+ 本地 Silero-VAD | `ml-inory/Her.axera` |
| TTS | 5000 | POST `/`、`/tts`（`{text, text_language}`→wav） | Her.axera `ax_tts`（Kokoro NPU，可回退 edge_tts） | `ml-inory/Her.axera` |
| RAG | 8002 | `/encode`、`/similarity`、`/ask` | bge-m3 AXMODEL（NPU，w8a16） | `AXERA-TECH/bge-m3` |
| BERT | 6007 | `/classify`（Vision / core memory） | Omni_fn_bert（Ernie-3.0-base）AXMODEL（NPU，INT8） | `morelle/Omni_fn_bert`（自编译） |

## 目录结构

```
axera/
├── asr_server.py           # ASR 适配服务（原 asr_api.py 契约）
├── tts_server.py           # TTS 适配服务（原 GPT-SoVITS api.py 契约）
├── rag_server.py           # RAG 适配服务（原 run_rag.py 契约）
├── bert_server.py          # BERT 适配服务（原 omni_bert_api.py 契约）
├── config.axera.json       # 前端配置样例（把 <BOARD_IP> 换成板子 IP 写入 live-2d/config.json）
├── requirements.txt        # 板端 Python 依赖
├── deps/                   # git submodule：sensevoice.axera / melotts.axera
├── models/
│   ├── export_ernie_onnx.py    # Omni_fn_bert -> 静态 ONNX（int32 输入 + NaN 防护清理）
│   ├── compile_axmodel.sh      # Pulsar2 编译 -> omni_fn_bert.axmodel
│   └── download_models.sh      # 全量模型下载（ModelScope/HF 镜像）
├── scripts/download_hf.sh      # hf-mirror hfd 多线程下载
└── deploy/
    ├── install_board.sh        # AX650 一键部署（NFS + 依赖 + 模型 + 启动）
    ├── start_all.sh            # 启动全部服务（nohup + pid）
    └── stop_all.sh
```

## 部署到 AX650 板（建议先读完整流程）

### 前置条件

- 一台 AX650 开发板（AX650N/AX650C，Ubuntu，4GB 内存），能 ssh root 登录
- 板端能挂载开发机的 NFS 共享（默认 `10.122.86.219:/data/shared` → `/mnt`），
  模型/依赖/日志全部放 NFS，避免占满板端根分区
- 开发机有 docker + `pulsar2:7.0` 镜像（仅编译 BERT axmodel 需要）

### 开发机一次性准备（可选，能省板端大量下载）

```bash
# 1) 把仓库同步到 NFS 共享（板端 /mnt/axera/my-neuro）
rsync -a --exclude 'axera/models/Qwen2.5-1.5B-Instruct' --exclude 'axera/models/bge-m3' \
  ./ /data/shared/axera/my-neuro/

# 2) 预置依赖仓库与模型（由脚本完成）
mkdir -p /data/shared/axera/deps /data/shared/axera/models
cd /data/shared/axera/deps
git clone --depth 1 https://github.com/ml-inory/sensevoice.axera.git
git clone --depth 1 https://github.com/ml-inory/melotts.axera.git
cd sensevoice.axera/python
HF_ENDPOINT=https://hf-mirror.com python3 -c \
  "from huggingface_hub import snapshot_download; snapshot_download('AXERA-TECH/SenseVoice', allow_patterns=['sensevoice_ax650/**'], local_dir='models/SenseVoice')"
cd ../../melotts.axera && bash download_models.sh

cd /data/shared/axera/models
bash <repo>/axera/models/download_models.sh     # vad + bert + bge-m3 + qwen

# 3) 板端一键安装
ssh root@<BOARD_IP> "cd /mnt/axera/my-neuro && bash axera/deploy/install_board.sh"
```

### 板端安装脚本做的事

1. 挂载 NFS 到 `/mnt`（`mount -t nfs4 <HOST>:/ /mnt -o nolock`）
2. `apt` 安装系统依赖（libsndfile / mecab / espeak-ng / cmake 等）
3. 链接 `axera/deps` → `/mnt/axera/deps`（sensevoice / melotts 及模型）
4. `pip --target` 安装 Python 依赖到 `/mnt/axera/pylib`（不占板端根分区，无需 apt/venv）
5. 下载/链接模型到 `/mnt/axera/models`（vad / bert / bge-m3）
6. 部署 Her.axera 后端到 `/mnt/axera/deps/Her.axera`，生成 `backend/.env`
   （ASR/TTS 用 ax_asr/ax_tts NPU；LLM 填 `DEEPSEEK_API_KEY` 或 `OPENAI_COMPAT_*`）
   - 安装 `ax_asr` / `ax_tts` aarch64 wheel（已预置 wheels 目录）
   - Kokoro 模型 + 中文音色 `zf_xiaoxiao` + espeak/jieba 词典预置到 `/mnt/axera/models`
7. （可选）`AXLLM_ENABLE=1` 安装 axllm + Qwen3-0.6B 作为端侧 LLM
8. `start_all.sh` 启动全部服务并输出访问地址

## 前端切换

把 [axera/config.axera.json](axera/config.axera.json) 里的 `<BOARD_IP>` 全部替换成
AX650 板的 IP，然后覆盖 `live-2d/config.json` 即可。前端其它配置（性格、UI 等）保持不变。
LLM 默认走 Her.axera 云端网关（`http://<BOARD_IP>:8080/v1`，OpenAI 兼容）。
云端凭据在 `/mnt/axera/deps/Her.axera/backend/.env` 配置：
DeepSeek 填 `DEEPSEEK_API_KEY`；其它 OpenAI 兼容服务把 `DEFAULT_LLM_PROVIDER=openai_compat`
并填 `OPENAI_COMPAT_API_BASE/API_KEY/MODEL`。
端侧 LLM（axllm+Qwen3-0.6B，约 1.1GB 权重）默认不部署，需要时 `AXLLM_ENABLE=1` 安装，
端口 8001（演示板 8000 常被其它服务占用）。

ASR/TTS 也走 Her.axera：`axera/asr_server.py` / `axera/tts_server.py` 只是前端契约的薄
适配层（`/v1/upload_audio`、`/v1/ws/vad`、POST `/` 转发到 Her.axera 的
`/v1/audio/transcriptions`、`/v1/audio/speech`）。RAG/BERT 保持本分支实现。

## BERT axmodel 复现编译（开发机）

```bash
# 1) 导出 ONNX（在 pulsar2 镜像里跑，自带 torch/transformers）
docker run --rm -v <bert权重目录>:/workspace/bert -v <导出目录>:/workspace/out \
  -v axera/models/export_ernie_onnx.py:/workspace/export_ernie_onnx.py pulsar2:7.0 \
  python3 /workspace/export_ernie_onnx.py --model /workspace/bert --out /workspace/out

# 2) 编译 AXMODEL
bash axera/models/compile_axmodel.sh <导出目录> <编译输出目录>
```

导出脚本自动处理了两个 Pulsar2 兼容性问题：
`int64 → int32` 输入转换，以及移除 transformers fp16 NaN 防护产生的 `IsNaN` 算子
（`Where(IsNaN(x), 0, x)` 在输入有限时等价于恒等，Pulsar2 不支持 IsNaN）。
编译结果 cosine = 1.0（compiler check 3 内置验证）。

## 已知限制

- LLM 默认云端（Her.axera 网关），端侧 axllm 为可选；Qwen3-0.6B 需约 1.1GB 权重
  常驻内存，4GB 板同时跑其它服务会比较紧
- TTS 为 MeloTTS 音色（与原 GPT-SoVITS 肥牛音色不同）；如需原音色，TTS 仍需在 PC 端跑 GPT-SoVITS
- ASR 热词通过 `hotwords.txt` 透传给 SenseVoice（原 funasr 的权重热词语法不适用）
- MemOS（`plugins-dlc/memos`）为纯 Python 记忆系统，未随本分支上板；其 embedding 可复用 RAG 服务
