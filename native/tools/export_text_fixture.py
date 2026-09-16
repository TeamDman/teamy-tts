"""Export exact deployed acoustic/PCM tensors for an arbitrary real text case."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
from export_model import write_tensors

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--upstream",type=Path,required=True)
p.add_argument("--models",type=Path,required=True)
p.add_argument("--text",required=True)
p.add_argument("--alpha",type=float,default=1.0)
p.add_argument("--voice",default="p2")
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
output=a.output.resolve();models=a.models.resolve();root=a.upstream.resolve()
os.chdir(root);sys.path.insert(0,str(root))
from utils.tools import prepare_text
tokens=prepare_text(a.text)
model=torch.jit.load(str(models/"glados-new.pt"),map_location="cpu").eval()
vocoder=torch.jit.load(str(models/"vocoder-gpu.pt"),map_location="cuda:0").eval()
import numpy as np
speaker=torch.from_numpy(np.fromfile(models/f"voice-{a.voice}.f32le",dtype="<f4").copy()).reshape(1,256)
with torch.inference_mode():
    out=model.generate_jit(tokens,speaker,a.alpha)
    audio=vocoder(out["mel_post"].cuda()).cpu()
    tensors={"tokens":tokens,"speaker":speaker,"mel":out["mel_post"],"audio":audio}
    tensors.update({f"acoustic.{key}":value for key,value in out.items() if isinstance(value,torch.Tensor)})
    write_tensors(output,tensors)
print(json.dumps({"text":a.text,"tokens":tokens.tolist(),"frames":out["mel_post"].shape[-1],"samples":audio.numel()},indent=2))
