#Prepare the VideoPose3D trained model for NNAPI on Android device
import sys
import os
import torch
import torch.utils.bundled_inputs
import torch.utils.mobile_optimizer
import torch.backends._nnapi.prepare
import torchvision.models.quantization.mobilenet
from pathlib import Path

from common.model import trace_model


# This script supports 3 modes of quantization:
# - "none": Fully floating-point model.
# - "core": Quantize the core of the model, but wrap it a
#    quantizer/dequantizer pair, so the interface uses floating point.
# - "full": Quantize the model, and use quantized tensors
#   for input and output.
#
# "none" maintains maximum accuracy
# "core" sacrifices some accuracy for performance,
# but maintains the same interface.
# "full" maximized performance (with the same accuracy as "core"),
# but requires the application to use quantized tensors.
#
# There is a fourth option, not supported by this script,
# where we include the quant/dequant steps as NNAPI operators.
def make_videopose3d_nnapi(output_dir_path, quantize_mode):
    quantize_core, quantize_iface = {
        "none": (False, False),
        "core": (True, False),
        "full": (True, True),
    }[quantize_mode]

    model, layers = trace_model() 


    # Fuse BatchNorm operators in the floating point model.
    # (Quantized models already have this done.)
    # Remove dropout for this inference-only use case.
    #if not quantize_core:
    #    model.fuse_model()



    
    #assert type(model.classifier[0]) == torch.nn.Dropout


    #model.classifier[0] = torch.nn.Identity()


    
    input_float = torch.zeros(2, 272, 17, 2)
    input_tensor = input_float

    
    # If we're doing a quantized model, we need to trace only the quantized core.
    # So capture the quantizer and dequantizer, use them to prepare the input,
    # and replace them with identity modules so we can trace without them.
    if quantize_core:
        print("quantize_core true")
        quantizer = model.quant
        dequantizer = model.dequant
        model.quant = torch.nn.Identity()
        model.dequant = torch.nn.Identity()
        input_tensor = quantizer(input_float)
    else:
        print("quantize_core false")


    
    # Many NNAPI backends prefer NHWC tensors, so convert our input to channels_last,
    # and set the "nnapi_nhwc" attribute for the converter.
    input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    input_tensor.nnapi_nhwc = True

    # Trace the model.  NNAPI conversion only works with TorchScript models,
    # and traced models are more likely to convert successfully than scripted.
    with torch.no_grad():
        traced = torch.jit.trace(model, input_tensor)
       
       
    #scripted = torch.jit.script(model)



    nnapi_model = torch.backends._nnapi.prepare.convert_model_to_nnapi(traced, input_tensor)

    '''
    # If we're not using a quantized interface, wrap a quant/dequant around the core.
    if quantize_core and not quantize_iface:
        nnapi_model = torch.nn.Sequential(quantizer, nnapi_model, dequantizer)
        model.quant = quantizer
        model.dequant = dequantizer
        # Switch back to float input for benchmarking.
        input_tensor = input_float.contiguous(memory_format=torch.channels_last)'''

    #Optimize the CPU model to make CPU-vs-NNAPI benchmarks fair.
    #model = torch.utils.mobile_optimizer.optimize_for_mobile(torch.jit.script(model))

    '''
    # Bundle sample inputs with the models for easier benchmarking.
    # This step is optional.
    class BundleWrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod
        def forward(self, arg):
            return self.mod(arg)


    nnapi_model = torch.jit.script(BundleWrapper(nnapi_model))
    torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
        model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])
    torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
        nnapi_model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])'''


    # Save both models.
    '''
    model.save(output_dir_path / ("videopose3d-quant_{}-cpu.pt".format(quantize_mode)))
    nnapi_model.save(output_dir_path / ("videopose3d-quant_{}-nnapi.pt".format(quantize_mode)))'''




if __name__ == "__main__":
    for quantize_mode in ["none", "core", "full"]:
        make_videopose3d_nnapi(Path(os.environ["HOME"]) / "mobilenetv2-nnapi", quantize_mode)