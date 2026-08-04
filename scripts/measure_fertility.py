"""Measure tokenizer fertility per language on a shared item set."""
import numpy as np
from transformers import AutoTokenizer

from mirage.data.loaders import load_benchmark, load_specs

spec = load_specs("configs/benchmarks.yaml")["global_mmlu"]
langs = ["en", "fr", "de", "ru", "zh", "ar", "hi", "bn", "te", "si", "am", "el", "sw", "yo"]
langs = [x for x in langs if x in spec.languages]
spec.max_items = 200
items, _ = load_benchmark(spec, languages=langs)

for name in ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
    tok = AutoTokenizer.from_pretrained(name)
    base = None
    print(f"\n=== {name} ===")
    for lg in langs:
        n = np.mean([len(tok.encode(it.prompt)) for it in items[lg]])
        if base is None:
            base = n
        print(f"  {lg:<4} {n:7.1f} tokens  fertility x{n / base:.2f}")
