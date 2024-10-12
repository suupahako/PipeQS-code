import quant_cuda as integer_quantizer
import torch
from torch import Tensor
from typing import Dict, List, Tuple, Union
import time
import gc




'''
*************************************************
***** quantization/dequantization functions *****
*************************************************
'''

# 这个函数的作用是计算输入张量的每行的最小值和最大值，并返回这些最小值和最大值。

# 让我们来逐步解释这个函数：

# input: Tensor：这是一个输入张量，其中每行代表一个样本，每列代表样本的特征。
# -> Tuple[Tensor, Tensor]：函数的返回类型是一个包含两个张量的元组。
# rmin, rmax = torch.min(input, dim=1)[0], torch.max(input, dim=1)[0]：
# 这行代码使用 PyTorch 的 min 和 max 函数计算输入张量 input 沿着第一个维度（即行）的最小值和最大值。这两个函数返回一个元组，其中第一个元素是最小值，第二个元素是对应的索引。我们在这里通过索引 [0] 只获取最小值本身，而不关心索引。rmin 存储了每行的最小值，rmax 存储了每行的最大值。
# return rmin, rmax：最后，将最小值和最大值作为元组返回。
# 这个函数通常用于归一化或标准化数据，以便在训练神经网络时提高训练效果

def compute_minmax_params(input: Tensor) -> Tuple[Tensor, Tensor]:
    rmin, rmax = torch.min(input, dim=1)[0], torch.max(input, dim=1)[0]
    return rmin, rmax


# 数据形状变化：
# 在输入之前，数据张量的形状为 [N, F]，其中 N 是批量大小，F 是特征维度。
# 在量化之后，数据张量的形状可能会发生变化，变成了 [N/(8/bits)*F] 的形状，其中 (8/bits) 是每个量化数所占的比特位数。
# 数据值变化：
# 在输入之前，数据张量的值是原始数据，可以是任意浮点数值。
# 在量化之后，数据张量的值被量化为整数，取值范围通常在 [0, 2^bits - 1] 之间，具体取值由量化函数和输入数据的范围决定。
# 整数值的范围根据 bits 参数确定，通常 bits 越大，可表示的数值范围越广，但精度会降低。
# 缩放因子变化：
# 在量化过程中，会计算每个节点的缩放因子 scale，用于将原始数据缩放到量化后的整数值范围内。
# scale 的形状与 rmin 和 rmax 相同，都是 [N]，其中 N 是数据批量大小。
# 缩放因子 scale 是一个浮点数张量，用于将原始数据缩放到量化后的整数值范围内，以确保量化后的数据保持尽可能接近原始数据的范围和分布。
# 综上所述，整数量化函数会将输入的浮点数数据张量转换为整数数据张量，并伴随着缩放因子的计算，以保证数据的尽可能准确的量化。

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
    # unpack bit stream
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

# def message_quantization(input: Tensor, bits: int, stochastic: bool) -> Tuple[Tensor, Tensor, Tensor, torch.Size]:
#     print_memory_usage("Start of message_quantization")

#     rmin, rmax = compute_minmax_params(input)
#     print_memory_usage("After compute_minmax_params")

#     q_input, q_scale = integer_quantize(input, bits, rmin, rmax, stochastic=stochastic)
#     print_memory_usage("After integer_quantize")

#     # transfer with bfloat16
#     if input.dtype == torch.float32:
#         return q_input, q_scale.to(torch.bfloat16), rmin.to(torch.bfloat16), input.shape
#     else:
#         return q_input, q_scale, rmin, input.shape

# def message_quantization(input: Tensor, bits: int, stochastic: bool, 
#                          q_input_holder: Tensor, q_scale_holder: Tensor, rmin_holder: Tensor) -> torch.Size:
#     rmin, rmax = compute_minmax_params(input)
#     q_input, q_scale = integer_quantize(input, bits, rmin, rmax, stochastic=stochastic)
    
#     # 将量化后的张量复制到指定的持有者中
#     q_input_holder.copy_(q_input)
#     q_scale_holder.copy_(q_scale)
#     rmin_holder.copy_(rmin)
    
#     # 如果需要转换数据类型，可以在此处转换
#     if input.dtype == torch.float32:
#         q_scale_holder.copy_(q_scale.to(torch.bfloat16))
#         rmin_holder.copy_(rmin.to(torch.bfloat16))
    
#     # 清除中间张量
#     del rmax, q_input, q_scale, rmin
    
#     return input.shape

# def message_quantization(input: Tensor, bits: int, stochastic: bool) -> Tuple[Tensor, Tensor, Tensor, torch.Size]:
#     for i in range(10000000):
#         print_memory_usage("Start of message_quantization")

#         rmin, rmax = compute_minmax_params(input)
#         print_memory_usage("After compute_minmax_params")

#         q_input, q_scale = integer_quantize(input, bits, rmin, rmax, stochastic=stochastic)
#         print_memory_usage("After integer_quantize")

#     # transfer with bfloat16
#     if input.dtype == torch.float32:
#         return q_input, q_scale.to(torch.bfloat16), rmin.to(torch.bfloat16), input.shape
#     else:
#         return q_input, q_scale, rmin, input.shape

def print_memory_usage(step_name):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"{step_name} - Allocated memory: {allocated:.2f} MB, Reserved memory: {reserved:.2f} MB")

# def message_quantization(input: torch.Tensor, bits: int, stochastic: bool, 
#                          q_input_holder: torch.Tensor, q_scale_holder: torch.Tensor, rmin_holder: torch.Tensor) -> torch.Size:
#     print_memory_usage("Start of message_quantization")
    
#     rmin, rmax = compute_minmax_params(input)
#     print_memory_usage("After compute_minmax_params")

#     q_input, q_scale = integer_quantize(input, bits, rmin, rmax, stochastic=stochastic)
#     print_memory_usage("After integer_quantize")

#     q_input_holder.copy_(q_input)
#     q_scale_holder.copy_(q_scale)
#     rmin_holder.copy_(rmin)
#     print_memory_usage("After copying to holders")

#     if input.dtype == torch.float32:
#         q_scale_holder.copy_(q_scale.to(torch.bfloat16))
#         rmin_holder.copy_(rmin.to(torch.bfloat16))
#         print_memory_usage("After converting to bfloat16")

#     # 删除中间变量
#     del rmax, q_input, q_scale, rmin
#     print_memory_usage("After deleting temporary variables")
#     # torch.cuda.empty_cache()
#     # print_memory_usage("After empty_cache variables")
#     torch.cuda.empty_cache()
#     print_memory_usage("After empty_cache variables")
#     return input.shape



# def message_dequantization(q_input: Tensor, q_scale: Tensor, rmin: Tensor, input_tempin_shape: torch.Size, bits):

#     # 确保传入的数据类型和形状正确
#     assert isinstance(q_input, torch.Tensor), "q_input should be a torch.Tensor"
#     assert isinstance(q_scale, torch.Tensor), "q_scale should be a torch.Tensor"
#     assert isinstance(rmin, torch.Tensor), "rmin should be a torch.Tensor"
#     assert isinstance(input_tempin_shape, tuple), "input_shape should be a tuple"
#     assert isinstance(bits, int), "bits should be an int"
#     if q_scale.dtype == torch.bfloat16:
#         q_scale = q_scale.to(torch.float32)
#         rmin = rmin.to(torch.float32)
#     input = integer_dequantize(q_input, input_tempin_shape, bits, q_scale, rmin)
#     return input.contiguous()

def message_dequantization(q_input: Tensor, q_scale: Tensor, rmin: Tensor, input_tempin_shape: torch.Size, bits):
    if q_scale.dtype == torch.bfloat16:
        q_scale = q_scale.to(torch.float32)
        rmin = rmin.to(torch.float32)
    input = integer_dequantize(q_input, input_tempin_shape, bits, q_scale, rmin)
    return input.contiguous()

# # 功能函数，用于打包和解包张量
# def pack_tensors(rmin, scale, shape, qt_f):
#     """
#     将四个张量打包为一个张量。
    
#     参数：
#     - rmin: torch.Tensor
#     - scale: torch.Tensor
#     - shape: torch.Tensor (必须为torch.int64类型)
#     - qt_f: torch.Tensor
    
#     返回：
#     - packed_tensor: 打包后的张量
#     """
#     # 确保shape是torch.int64类型
#     assert shape.dtype == torch.int64, "shape tensor must be of type torch.int64"
    
#     # 将所有张量展平
#     rmin_flat = rmin.flatten()
#     scale_flat = scale.flatten()
#     shape_flat = shape.flatten()
#     qt_f_flat = qt_f.flatten()
    
#     # 获取每个张量的大小
#     sizes = torch.tensor([len(rmin_flat), len(scale_flat), len(shape_flat), len(qt_f_flat)], dtype=torch.int64)
    
#     # 打包大小信息和张量数据
#     packed_tensor = torch.cat([sizes, rmin_flat, scale_flat, shape_flat, qt_f_flat])
    
#     return packed_tensor

# def unpack_tensors(packed_tensor):
#     """
#     将一个打包张量解包为四个张量。
    
#     参数：
#     - packed_tensor: 打包后的张量
    
#     返回：
#     - rmin: torch.Tensor
#     - scale: torch.Tensor
#     - shape: torch.Tensor (torch.int64类型)
#     - qt_f: torch.Tensor
#     """
#     # 获取每个张量的大小信息
#     sizes = packed_tensor[:4]
#     rmin_size, scale_size, shape_size, qt_f_size = sizes.tolist()
    
#     # 解包张量数据
#     rmin_flat = packed_tensor[4:4 + rmin_size]
#     scale_flat = packed_tensor[4 + rmin_size:4 + rmin_size + scale_size]
#     shape_flat = packed_tensor[4 + rmin_size + scale_size:4 + rmin_size + scale_size + shape_size]
#     qt_f_flat = packed_tensor[4 + rmin_size + scale_size + shape_size:]
    
#     # 恢复原始形状
#     rmin = rmin_flat.view(-1)
#     scale = scale_flat.view(-1)
#     shape = shape_flat.view(2)
#     qt_f = qt_f_flat.view(-1)
    
#     return rmin, scale, shape, qt_f

# def pack_tensors(rmin, scale, shape, qt_f):
#     assert qt_f.dtype == torch.int8
#     assert scale.dtype == torch.float32
#     assert rmin.dtype == torch.float32
#     assert shape.dtype == torch.int64
    
#     qt_f_flat = qt_f.view(-1).float()
#     sizes = torch.tensor([qt_f_flat.numel(), scale.numel(), rmin.numel(), shape.numel()], dtype=torch.int64, device=qt_f.device)
#     packed_tensor = torch.cat([qt_f_flat, scale, rmin, shape.float(), sizes.float()], dim=0)
#     # DEBUG
#     print(f"pack_tensors in init buffer: {packed_tensor}")
#     return packed_tensor

def pack_tensors(rmin, scale, shape, qt_f):
    assert qt_f.dtype == torch.int8, f"Expected int8 for qt_f but got {qt_f.dtype}"
    assert scale.dtype == torch.bfloat16, f"Expected bfloat16 for scale but got {scale.dtype}"
    assert rmin.dtype == torch.bfloat16, f"Expected bfloat16 for rmin but got {rmin.dtype}"
    assert shape.dtype == torch.int64, f"Expected int64 for shape but got {shape.dtype}"
    
    # 确保所有张量在同一设备上
    device = qt_f.device
    # scale = scale.to(device)
    # rmin = rmin.to(device)
    shape = shape.to(device)

    # 计算各个张量的大小
    sizes = torch.tensor([qt_f.numel(), scale.numel(), rmin.numel(), shape.numel()], dtype=torch.int64, device=device)
    
    # 打包所有张量，包括大小信息
    packed_tensor = torch.cat([qt_f, scale, rmin, shape.float(), sizes.float()], dim=0)

    
    return packed_tensor


def pack_tensors_in_place(destination, rmin, scale, shape, qt_f):
    assert qt_f.dtype == torch.int8, f"Expected int8 for qt_f but got {qt_f.dtype}"
    assert scale.dtype == torch.bfloat16, f"Expected bfloat16 for scale but got {scale.dtype}"
    assert rmin.dtype == torch.bfloat16, f"Expected bfloat16 for rmin but got {rmin.dtype}"
    assert shape.dtype == torch.int64, f"Expected int64 for shape but got {shape.dtype}"

    # 将所有张量移动到CPU，确保 destination 是在 CPU 上的
    qt_f = qt_f.cpu()
    scale = scale.cpu()
    rmin = rmin.cpu()
    shape = shape.cpu().float()

    # 计算各个张量的大小
    qt_f_size = qt_f.numel()
    scale_size = scale.numel()
    rmin_size = rmin.numel()
    shape_size = shape.numel()

    # 填充目标张量 destination
    destination[0:qt_f_size] = qt_f
    destination[qt_f_size:qt_f_size + scale_size] = scale
    destination[qt_f_size + scale_size:qt_f_size + scale_size + rmin_size] = rmin
    destination[qt_f_size + scale_size + rmin_size:qt_f_size + scale_size + rmin_size + shape_size] = shape

    # 添加 sizes 信息
    sizes = torch.tensor([qt_f_size, scale_size, rmin_size, shape_size], dtype=torch.float32)
    destination[-4:] = sizes

# def pack_tensors_in_place(destination, rmin, scale, shape, qt_f):
#     assert qt_f.dtype == torch.int8, f"Expected int8 for qt_f but got {qt_f.dtype}"
#     assert scale.dtype == torch.bfloat16, f"Expected bfloat16 for scale but got {scale.dtype}"
#     assert rmin.dtype == torch.bfloat16, f"Expected bfloat16 for rmin but got {rmin.dtype}"
#     assert shape.dtype == torch.int64, f"Expected int64 for shape but got {shape.dtype}"
    
#     # 计算各个张量的大小
#     qt_f_size = qt_f.numel()
#     scale_size = scale.numel()
#     rmin_size = rmin.numel()
#     shape_size = shape.numel()

#     # 填充目标张量
#     destination[0:qt_f_size] = qt_f
#     destination[qt_f_size:qt_f_size + scale_size] = scale
#     destination[qt_f_size + scale_size:qt_f_size + scale_size + rmin_size] = rmin
#     destination[qt_f_size + scale_size + rmin_size:qt_f_size + scale_size + rmin_size + shape_size] = shape.float()

#     # 将 sizes 信息放在最后
#     sizes = torch.tensor([qt_f_size, scale_size, rmin_size, shape_size], dtype=torch.float32, device=qt_f.device)
#     destination[-4:] = sizes

#     return destination


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
    
    # DEBUG
    # print(f"unpacked qt_f: {qt_f}")
    # print(f"unpacked scale: {scale}")
    # print(f"unpacked rmin: {rmin}")
    # print(f"unpacked shape: {shape}")
    # print(f"sizes: {sizes}")
    
    return rmin, scale, shape, qt_f

def pack_param_tensors(rmin, scale, shape):
    assert scale.dtype == torch.bfloat16, f"Expected bfloat16 for scale but got {scale.dtype}"
    assert rmin.dtype == torch.bfloat16, f"Expected bfloat16 for rmin but got {rmin.dtype}"
    assert shape.dtype == torch.int64, f"Expected int64 for shape but got {shape.dtype}"
    
    # 确保所有张量在同一设备上
    device = rmin.device
    scale = scale.to(device)
    rmin = rmin.to(device)
    shape = shape.to(device)
    
    # 计算各个张量的大小
    sizes = torch.tensor([rmin.numel(), scale.numel(), shape.numel()], dtype=torch.int64, device=device)
    
    # 打包所有张量，包括大小信息
    packed_tensor = torch.cat([rmin.float(), scale.float(), shape.float(), sizes.float()], dim=0)
    
    return packed_tensor

def unpack_param_tensors(packed_tensor):
    # 从打包的张量中提取大小信息
    sizes = packed_tensor[-3:].to(torch.int64)
    rmin_size, scale_size, shape_size = sizes.tolist()
    
    rmin_end = rmin_size
    scale_end = rmin_end + scale_size
    shape_end = scale_end + shape_size
    
    rmin = packed_tensor[:rmin_end].to(torch.bfloat16)
    scale = packed_tensor[rmin_end:scale_end].to(torch.bfloat16)
    shape = packed_tensor[scale_end:shape_end].to(torch.int64)
    
    return rmin, scale, shape
