"""Translate a request once, before loading the exploration models."""

import gc
import json
from pathlib import Path


MODEL = "Qwen/Qwen3-1.7B"
CACHE_DIR = Path(__file__).resolve().parents[3] / "models/huggingface"
PROMPT = """Translate the user's robot search request into concrete English object labels
for an open-vocabulary detector. Return only JSON: {"labels": ["chair", "sofa"]}.
Use at most 8 short labels. Preserve explicit attributes such as color.
For a named object, return only that object. For a function, return objects that
provide that function, not associated accessories. Examples:
"a place to sit" -> {"labels": ["chair", "sofa", "bench", "stool"]}
"a surface to write on" -> {"labels": ["desk", "table"]}
"red backpack" -> {"labels": ["red backpack"]}
"somewhere to sleep" -> {"labels": ["bed", "sofa"]}
"""


def parse_labels(response):
    """Accept a bounded label list; reject malformed model output."""
    response = response.strip().removeprefix("```json").removeprefix("```")
    response = response.removesuffix("```").strip()
    try:
        data = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("Qwen did not return valid label JSON") from error
    labels = data.get("labels") if isinstance(data, dict) else None
    if (not isinstance(labels, list) or not 1 <= len(labels) <= 8
            or any(not isinstance(label, str) or not label.strip() or len(label) > 80
                   for label in labels)):
        raise ValueError("Qwen must return 1–8 nonempty, short object labels")
    return tuple(dict.fromkeys(label.strip() for label in labels))


def translate_goal(request, device="cpu"):
    """Run Qwen3-1.7B once and release it, including on generation failure."""
    if not request.strip():
        raise ValueError("--object-query must contain a request")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--query-device cuda requires an available CUDA device")
    model = tokenizer = inputs = generated = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, cache_dir=CACHE_DIR, dtype="auto").to(device).eval()
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": PROMPT},
             {"role": "user", "content": request.strip()}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        return parse_labels(response)
    finally:
        del model, tokenizer, inputs, generated
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
