import torch
import torchvision

from common.model import trace_model


m, layers = trace_model()

for mod in m.layers_conv:
    print("This mod is", mod)


#print(m.modules())


#working on fusing the pairs of conv
#f = torch.quantization.fuse_modules(m, layers, inplace=False)


#lighten the model by converting to 8-bit ints
types_to_quantize = {torch.nn.Conv1d, torch.nn.BatchNorm1d, torch.nn.ReLU}
q = torch.quantization.quantize_dynamic(m, types_to_quantize, dtype=torch.qint8)

s = torch.jit.script(q)
torch.jit.save(s, "processed_mod.pt")


