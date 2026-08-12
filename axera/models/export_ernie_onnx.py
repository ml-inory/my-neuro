#!/usr/bin/env python3
"""导出 morelle/Omni_fn_bert（Ernie-3.0-base-zh 二分类）为静态 ONNX。

在 pulsar2 镜像内执行（自带 torch/transformers/onnx/onnxruntime）：
  docker run --rm -v /tmp/bert_model:/workspace/bert -v <out>:/workspace/out pulsar2:7.0 \
    python3 /workspace/export_ernie_onnx.py --model /workspace/bert --out /workspace/out

产物：
  <out>/model.onnx           静态 shape (1,512)，int32 外部输入
  <out>/model_meta.json      输入/输出契约
  <out>/calib/<input>/*.npy  真实业务校准数据（Pulsar2 input_configs 用）
  <out>/calib/<input>.tar.gz
"""
import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def ensure_int32_inputs(onnx_path: Path):
    """把 int64 外部输入改为 int32 + 入口 Cast（Pulsar2 需要 int32）。"""
    model = onnx.load(str(onnx_path))
    graph = model.graph
    inserts = []
    for inp in list(graph.input):
        tt = inp.type.tensor_type
        if tt.elem_type == onnx.TensorProto.INT64:
            orig = inp.name
            internal = f"{orig}_i64"
            shape = [d.dim_value for d in tt.shape.dim]
            new_ext = onnx.helper.make_tensor_value_info(orig, onnx.TensorProto.INT32, shape)
            inp.name = internal
            cast = onnx.helper.make_node("Cast", [orig], [internal], to=onnx.TensorProto.INT64, name=f"cast_{orig}")
            inserts.append((new_ext, cast))
    for new_ext, cast in inserts:
        graph.input.insert(0, new_ext)
        graph.node.insert(0, cast)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    print(f"[export] int64 输入已转换为 int32 + Cast: {[e.name for e, _ in inserts]}")


def sanitize_nan_guard(onnx_path: Path):
    """Pulsar2 不支持 IsNaN：把 'Where(IsNaN(Softmax), 0, Softmax)' 替换为 Identity。

    transformers 的 fp16 NaN 防护（torch.where(torch.isnan(p), 0, p)），
    在输入有限时是恒等变换，删除后推理结果不变。
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    isnan_out = {n.output[0] for n in graph.node if n.op_type == "IsNaN"}
    removed = 0
    new_nodes = []
    for n in graph.node:
        if n.op_type == "IsNaN":
            removed += 1
            continue
        if n.op_type == "Where" and n.input[0] in isnan_out:
            ident = onnx.helper.make_node("Identity", [n.input[2]], list(n.output), name=f"{n.name}_id")
            new_nodes.append(ident)
            removed += 1
            continue
        new_nodes.append(n)
    del graph.node[:]
    graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    print(f"[export] 已移除 {removed} 个 IsNaN/Where NaN 防护子图")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型目录（含 config.json / model.safetensors）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--calib-samples", type=int, default=16)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model_dir = Path(args.model)
    out = Path(args.out)
    (out / "calib").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    # 真实业务风格的中文校准文本
    texts = [
        "你刚才说的话是什么意思",
        "我今天心情很好，想出去走走",
        "帮我记住我的生日是三月十五号",
        "请把这段对话总结一下",
        "你能看到我的屏幕吗",
        "我讨厌下雨天，路上全是水",
        "以后每天八点提醒我吃药",
        "你叫什么名字，为什么这么可爱",
        "这张图片里有什么东西",
        "把我刚刚说的内容存到记忆里",
        "我不太喜欢你开这种玩笑",
        "下次见面我们一起去吃饭吧",
        "这个功能怎么用，教教我",
        "你的性格可以调整吗",
        "帮我查一下明天的天气",
        "刚才那首歌叫什么名字",
    ][: args.calib_samples]

    enc = tokenizer(
        texts,
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(torch.int32)
    attention_mask = enc["attention_mask"].to(torch.int32)
    token_type_ids = enc["token_type_ids"].to(torch.int32)
    position_ids = torch.arange(args.max_length, dtype=torch.int32).unsqueeze(0).repeat(len(texts), 1)
    task_type_id = torch.zeros(len(texts), dtype=torch.int32)

    # forward 位置参数顺序（transformers 5.x ErnieForSequenceClassification）：
    # input_ids, attention_mask, token_type_ids, task_type_ids, position_ids
    example = (
        input_ids[:1],
        attention_mask[:1],
        token_type_ids[:1],
        task_type_id[:1],
        position_ids[:1],
    )
    input_names = ["input_ids", "attention_mask", "token_type_ids", "task_type_id", "position_ids"]
    output_names = ["logits"]

    with torch.no_grad():
        ref = model(*example).logits.detach().numpy().astype(np.float32)

    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
    )
    ensure_int32_inputs(onnx_path)
    sanitize_nan_guard(onnx_path)

    # ORT 对分（int32 feed）
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {
        "input_ids": input_ids[:1].numpy(),
        "attention_mask": attention_mask[:1].numpy(),
        "token_type_ids": token_type_ids[:1].numpy(),
        "task_type_id": task_type_id[:1].numpy(),
        "position_ids": position_ids[:1].numpy(),
    }
    ort_out = sess.run(None, feed)[0].astype(np.float32)
    cos = cosine(ref, ort_out)
    print(f"[export] Torch-ONNX cosine={cos:.6f}")
    if cos < 0.99:
        raise SystemExit(f"cosine {cos:.6f} < 0.99，导出失败")

    # 静态 shape 检查
    proto = onnx.load(str(onnx_path))
    meta_inputs = []
    for vi in proto.graph.input:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        meta_inputs.append({"name": vi.name, "shape": dims, "dtype": "int32"})
    meta_outputs = [{"name": vi.name, "shape": [d.dim_value for d in vi.type.tensor_type.shape.dim]} for vi in proto.graph.output]
    (out / "model_meta.json").write_text(
        json.dumps(
            {
                "model_name": "Omni_fn_bert",
                "framework": "pytorch",
                "inputs": meta_inputs,
                "outputs": meta_outputs,
                "opset": args.opset,
                "torch_onnx_cosine": cos,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 校准数据：每个输入一个目录 + tar.gz（与 bge-m3 参考一致）
    samples = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_type_ids": token_type_ids.numpy(),
        "task_type_id": task_type_id.numpy(),
        "position_ids": position_ids.numpy(),
    }
    for name, arr in samples.items():
        d = out / "calib" / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(arr.shape[0]):
            # Pulsar2 要求校准样本 shape 与模型输入完全一致（保留 batch 维）
            sample = arr[i : i + 1] if arr[i].ndim > 0 else np.expand_dims(arr[i], 0)
            np.save(d / f"{i:04d}.npy", np.ascontiguousarray(sample))
        tar_path = out / "calib" / f"{name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for npy in sorted(d.glob("*.npy")):
                tar.add(npy, arcname=npy.name)
    print(f"[export] OK: {onnx_path}")
    print(f"[export] 校准数据: {out / 'calib'}")


if __name__ == "__main__":
    main()
