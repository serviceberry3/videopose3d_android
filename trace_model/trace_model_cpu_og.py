import torch
import torchvision
import torch.utils.mobile_optimizer as mobile_optimizer

from common.model import trace_model


m, layers = trace_model()

#for mod in m.layers_conv:#
    #print("This mod is", mod)


#print(m.modules())



#working on fusing the pairs of conv
f = torch.quantization.fuse_modules(m, ['layers_conv.0', 'layers_bn.0'], inplace=False)
f1 = torch.quantization.fuse_modules(f, ['layers_conv.1', 'layers_bn.1'], inplace=False)
f2 = torch.quantization.fuse_modules(f1, ['layers_conv.2', 'layers_bn.2'], inplace=False)
f3 = torch.quantization.fuse_modules(f2, ['layers_conv.3', 'layers_bn.3'], inplace=False)
f4 = torch.quantization.fuse_modules(f3, ['layers_conv.4', 'layers_bn.4'], inplace=False)
f5 = torch.quantization.fuse_modules(f4, ['layers_conv.5', 'layers_bn.5'], inplace=False)
f6 = torch.quantization.fuse_modules(f5, ['layers_conv.6', 'layers_bn.6'], inplace=False)
f7 = torch.quantization.fuse_modules(f6, ['layers_conv.7', 'layers_bn.7'], inplace=False)
#f8 = torch.quantization.fuse_modules(f7, ['shrink', 'expand_bn'], inplace=False)

#f7 = torch.quantization.fuse_modules(m, ['layers_conv.0', 'layers_bn.0', 'layers_conv.1', 'layers_bn.1'], inplace=False)

for mod in f7.modules():
    print("This mod is", mod)


'''
# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
m.qconfig = torch.quantization.default_qconfig
print(m.qconfig)
torch.quantization.prepare(m, inplace=True)'''




#lighten the model by converting to 8-bit ints
types_to_quantize = {torch.nn.Conv1d, torch.nn.BatchNorm1d, torch.nn.ReLU}
q = torch.quantization.quantize_dynamic(f7, types_to_quantize, dtype=torch.qint8)


s = torch.jit.script(q)

mobile_optimized = mobile_optimizer.optimize_for_mobile(s)

#store the scripted model in parent directory of this repo
torch.jit.save(mobile_optimized, "../../processed_mod.pt")