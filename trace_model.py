import torch
import torchvision

from common.model import trace_model


m, layers = trace_model()
f = torch.quantization.fuse_modules(m, layers, inplace=False)

#lighten the model by converting to 8-bit ints
types_to_quantize = {torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU}
q = torch.quantization.quantize_dynamic(f, types_to_quantize, dtype=torch.qint8)

s = torch.jit.script(q)
torch.jit.save(s, "processed_mod.pt")


