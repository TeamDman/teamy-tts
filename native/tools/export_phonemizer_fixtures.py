"""Independent neural frontend oracle from the upstream DeepPhonemizer API."""
import argparse
import json
from pathlib import Path
import torch
from dp.phonemizer import Phonemizer
from export_model import write_tensors, fingerprint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
phonemizer = Phonemizer.from_checkpoint(str(args.checkpoint))
model = phonemizer.predictor.model.cpu().eval()
symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß'"
words = ["supercalifragilistic", "quux", "glados", "electroencephalographically", "a", "don't"]
records = []
with torch.inference_mode():
    for i, word in enumerate(words):
        tokens = [2] + [symbols.index(c) + 4 for c in word for _ in range(3)] + [3]
        x = torch.tensor([tokens], dtype=torch.long)
        logits = model({"text": x})
        indices = logits.argmax(-1)
        file = f"phonemizer-{i}.safetensors"
        write_tensors(args.output / file, {"tokens": x, "logits": logits, "indices": indices})
        records.append({"word": word, "fixture": file, "indices": indices.tolist()})
receipt = {"checkpoint": fingerprint(args.checkpoint), "torch": torch.__version__, "fixtures": records}
(args.output / "phonemizer-fixtures.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
