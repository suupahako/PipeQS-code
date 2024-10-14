import quant_cuda as integer_quantizer
import torch
from torch import Tensor
from typing import Dict, List, Tuple, Union
import time
import gc

def compute_minmax_params(input: Tensor) -> Tuple[Tensor, Tensor]:
    rmin, rmax = torch.min(input, dim=1)[0], torch.max(input, dim=1)[0]
    return rmin, rmax

def integer_quantize(data: Tensor, bits: int, rmin: Tensor, rmax: Tensor, stochastic: bool = True) -> Tuple[Tensor, Tensor]:
    '''
    `input`
        data: shape: [N, F], where N is the batch_size, F is the feature dimension.

        bits: type: int, quantization bit width.

        rmin: shape: [N], min value per node, serve as zero point.

        rmax: shape: [N], max value per node.
    `return`
        q_data: [N/(8/bits)*F]

        scale: [N]
    '''
    assert type(bits) == int
    quant_func = integer_quantizer.pack_single_precision
    scale = (2 ** bits - 1) / (rmax - rmin)  # shape: [N]
    q_data = quant_func(data, rmin, rmax, scale.to(data.dtype), bits, stochastic)
    return q_data, scale

def integer_dequantize(q_data: Tensor, shape: torch.Size, bits: int, scale: Tensor, rmin: Tensor) -> Tensor:
    r'''
    input
        data: shape: [N/(8/bits)*F], where N is the batch_size, bits is the quantization bits,  F is the feature dimension. (already on device)

        shape: the tempinal shape of q_data

        bits: type: int, quantization bit width.

        scale: shape: [N], quantization scale per node. (already on device)

        rmin: shape: [N], min value per node, serve as zero point.

    return
        data: shape: [N, F], where N is the batch_size, F is the feature dimension.
    '''
    N = shape[0]
    num_features = shape[1]
    assert type(bits) == int
    dequant_func = integer_quantizer.unpack_single_precision
    data = dequant_func(q_data, bits, scale, rmin, N, num_features)
    return data



def message_quantization(input: Tensor, bits: int, stochastic: bool) -> Tuple[Tensor, Tensor, Tensor, torch.Size]:
    rmin, rmax = compute_minmax_params(input)
    q_input, q_scale = integer_quantize(input, bits, rmin, rmax, stochastic=stochastic)
    # transfer with bfloat16
    if input.dtype == torch.float32:
        return q_input, q_scale.to(torch.bfloat16), rmin.to(torch.bfloat16), input.shape
    else:
        return q_input, q_scale, rmin, input.shape


def print_memory_usage(step_name):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"{step_name} - Allocated memory: {allocated:.2f} MB, Reserved memory: {reserved:.2f} MB")


def message_dequantization(q_input: Tensor, q_scale: Tensor, rmin: Tensor, input_tempin_shape: torch.Size, bits):
    if q_scale.dtype == torch.bfloat16:
        q_scale = q_scale.to(torch.float32)
        rmin = rmin.to(torch.float32)
    input = integer_dequantize(q_input, input_tempin_shape, bits, q_scale, rmin)
    return input.contiguous()

def pack_tensors(rmin, scale, shape, qt_f):
    assert qt_f.dtype == torch.int8, f"Expected int8 for qt_f but got {qt_f.dtype}"
    assert scale.dtype == torch.bfloat16, f"Expected bfloat16 for scale but got {scale.dtype}"
    assert rmin.dtype == torch.bfloat16, f"Expected bfloat16 for rmin but got {rmin.dtype}"
    assert shape.dtype == torch.int64, f"Expected int64 for shape but got {shape.dtype}"
    
    device = qt_f.device
    shape = shape.to(device)
    sizes = torch.tensor([qt_f.numel(), scale.numel(), rmin.numel(), shape.numel()], dtype=torch.int64, device=device)
    packed_tensor = torch.cat([qt_f, scale, rmin, shape.float(), sizes.float()], dim=0)

    
    return packed_tensor


def pack_tensors_in_place(destination, rmin, scale, shape, qt_f):
    assert qt_f.dtype == torch.int8, f"Expected int8 for qt_f but got {qt_f.dtype}"
    assert scale.dtype == torch.bfloat16, f"Expected bfloat16 for scale but got {scale.dtype}"
    assert rmin.dtype == torch.bfloat16, f"Expected bfloat16 for rmin but got {rmin.dtype}"
    assert shape.dtype == torch.int64, f"Expected int64 for shape but got {shape.dtype}"

    qt_f = qt_f.cpu()
    scale = scale.cpu()
    rmin = rmin.cpu()
    shape = shape.cpu().float()

    qt_f_size = qt_f.numel()
    scale_size = scale.numel()
    rmin_size = rmin.numel()
    shape_size = shape.numel()

    destination[0:qt_f_size] = qt_f
    destination[qt_f_size:qt_f_size + scale_size] = scale
    destination[qt_f_size + scale_size:qt_f_size + scale_size + rmin_size] = rmin
    destination[qt_f_size + scale_size + rmin_size:qt_f_size + scale_size + rmin_size + shape_size] = shape

    sizes = torch.tensor([qt_f_size, scale_size, rmin_size, shape_size], dtype=torch.float32)
    destination[-4:] = sizes



def unpack_tensors(packed_tensor):
    sizes = packed_tensor[-4:].to(torch.int64)
    qt_f_size, scale_size, rmin_size, shape_size = sizes.tolist()
    
    qt_f_end = qt_f_size
    scale_end = qt_f_end + scale_size
    rmin_end = scale_end + rmin_size
    shape_end = rmin_end + shape_size
    
    qt_f = packed_tensor[:qt_f_end].to(torch.int8)
    scale = packed_tensor[qt_f_end:scale_end].to(torch.bfloat16)
    rmin = packed_tensor[scale_end:rmin_end].to(torch.bfloat16)
    shape = packed_tensor[rmin_end:shape_end].to(torch.int64)
    
    return rmin, scale, shape, qt_f

def pack_param_tensors(rmin, scale, shape):
    assert scale.dtype == torch.bfloat16, f"Expected bfloat16 for scale but got {scale.dtype}"
    assert rmin.dtype == torch.bfloat16, f"Expected bfloat16 for rmin but got {rmin.dtype}"
    assert shape.dtype == torch.int64, f"Expected int64 for shape but got {shape.dtype}"
    
    device = rmin.device
    scale = scale.to(device)
    rmin = rmin.to(device)
    shape = shape.to(device)
    
    sizes = torch.tensor([rmin.numel(), scale.numel(), shape.numel()], dtype=torch.int64, device=device)
    
    packed_tensor = torch.cat([rmin.float(), scale.float(), shape.float(), sizes.float()], dim=0)
    
    return packed_tensor

def unpack_param_tensors(packed_tensor):
    sizes = packed_tensor[-3:].to(torch.int64)
    rmin_size, scale_size, shape_size = sizes.tolist()
    
    rmin_end = rmin_size
    scale_end = rmin_end + scale_size
    shape_end = scale_end + shape_size
    
    rmin = packed_tensor[:rmin_end].to(torch.bfloat16)
    scale = packed_tensor[rmin_end:scale_end].to(torch.bfloat16)
    shape = packed_tensor[scale_end:shape_end].to(torch.int64)
    
    return rmin, scale, shape
