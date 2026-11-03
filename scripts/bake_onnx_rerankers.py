"""Export ONNX rerankers into FASTEMBED_CACHE_PATH/rerankers during image build.

optimum and torch are installed only for this step. The runtime image does not
include them and does not download weights.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(os.environ.get("FASTEMBED_CACHE_PATH", "/app/models")) / "rerankers"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "sentencepiece.bpe.model",
)


def save_tokenizer(model_id: str, dest: Path) -> None:
    """Write tokenizer.json next to the ONNX file.

    ORTModel.save_pretrained writes the graph only. Newer transformers builds
    leave the tokenizer behind, and a slow XLM-R tokenizer has no tokenizer.json
    until it is saved as the fast tokenizer.
    """
    from transformers import AutoTokenizer

    dest.mkdir(parents=True, exist_ok=True)
    try:
        AutoTokenizer.from_pretrained(model_id, use_fast=True).save_pretrained(dest)
    except (OSError, ValueError):
        pass
    if (dest / "tokenizer.json").is_file():
        return
    from transformers import XLMRobertaTokenizerFast

    XLMRobertaTokenizerFast.from_pretrained(model_id).save_pretrained(dest)
    if not (dest / "tokenizer.json").is_file():
        raise SystemExit(f"bake failed, tokenizer.json missing for {model_id}")


def export_fp32(model_id: str, dest: Path) -> None:
    from optimum.onnxruntime import ORTModelForSequenceClassification

    dest.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(dest)
    save_tokenizer(model_id, dest)


def quantize_int8(src: Path, dest: Path) -> None:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    dest.mkdir(parents=True, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(src)
    config = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=dest, quantization_config=config)
    for filename in TOKENIZER_FILES:
        source = src / filename
        if source.is_file() and not (dest / filename).is_file():
            shutil.copy(source, dest / filename)


def _require(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"bake failed, missing {path}")


def export_gte(model_id: str, dest: Path) -> None:
    """Export Alibaba-NLP/gte-multilingual-reranker-base.

    The checkpoint is a custom NewForSequenceClassification (RoPE, trust_remote_code),
    so optimum's stock sequence-classification exporter does not know it. The graph
    is the pair-scoring logits used by the model card: input_ids + attention_mask.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.float()
    model.eval()
    model.config.unpad_inputs = False
    model.config.use_memory_efficient_attention = False

    class _Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            out = self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return out.logits if hasattr(out, "logits") else out[0]

    wrapper = _Logits(model)
    wrapper.eval()
    dummy = tokenizer(
        [["warmup query", "warmup document"]],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    export_kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    try:
        torch.onnx.export(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(dest / "model.onnx"),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(dest / "model.onnx"),
            **export_kwargs,
        )
    save_tokenizer(model_id, dest)


def _finish_quantized(fp32: Path, dest: Path) -> None:
    quantize_int8(fp32, dest)
    quantized = dest / "model_quantized.onnx"
    if not quantized.is_file():
        exported = dest / "model.onnx"
        if exported.is_file():
            exported.rename(quantized)
    leftover = dest / "model.onnx"
    if quantized.is_file() and leftover.is_file():
        leftover.unlink()
    _require(quantized)
    _require(dest / "tokenizer.json")
    shutil.rmtree(fp32, ignore_errors=True)


def main() -> None:
    mmarco = ROOT / "mmarco"
    print("Exporting cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 to ONNX...")
    export_fp32("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", mmarco)
    _require(mmarco / "model.onnx")
    _require(mmarco / "tokenizer.json")

    fp32 = Path("/tmp/gte_reranker_fp32")
    print("Exporting Alibaba-NLP/gte-multilingual-reranker-base and quantizing to int8...")
    export_gte("Alibaba-NLP/gte-multilingual-reranker-base", fp32)
    _finish_quantized(fp32, ROOT / "gte")
    print("ONNX reranker bake complete.")


if __name__ == "__main__":
    main()
