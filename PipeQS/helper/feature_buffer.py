import torch
from multiprocessing.pool import ThreadPool
import torch.nn.functional as F
from multiprocessing import Event
from helper.timer.timer import *
from helper.quantization import *
from time import sleep
import queue
import io
import sys
import time
from config import *
import hashlib
import random
FULL_PRECISION = 1
BIT2_PRECISION = 2
BIT4_PRECISION = 3
BIT8_PRECISION = 4

# 你的代码

class Buffer(object):

    # 用于计算压缩后Tensor的大小，以生成缓冲区
    def calculate_packed_tensor_dimension(self, N, F, bits):
        # 每个线程处理的元素数
        work_per_thread = 8 // bits
        # 调整 N 使其成为 work_per_thread 的倍数
        N_round = N + (work_per_thread - N % work_per_thread) % work_per_thread
        # 总的位数
        total_bits = bits * (N_round * F)
        # 总的字节数
        total_bytes = 1 + (total_bits + 8 - 1) // 8  # 等价于 math.ceil(total_ bits / 8)
        return total_bytes

    def __init__(self):
        super(Buffer, self).__init__()
        self._num_in = None
        self._boundary = []
        self._n_layers = 0
        self._layer_size = []
        # 流水线
        self._pipeline = False
        # 记录epoch
        self._epoch = 0
        self._feat_cpu, self._grad_cpu = [], []
        # 注意，了解其结构和作用
        self._f_buf = []
        # 值得注意的是，PipeGCN中这类buffer变量，都有三层
        self._f_recv, self._b_recv = [], []
        self._f_recv_cpu, self._b_recv_cpu = [], []
        self._recv_shape = []
        # 线程池
        self._pool = None
        # 通讯流
        self._comm_stream = None
        # 特征和梯度的cpu、cuda事件
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []
        self._backend = None
        
        self._pl, self._pr = [], []
        # 新增内容
        # 新增内容，用于存储min
        self._rmin_recv = []
        # 反向传播中的rmin
        self._g_rmin_recv = []
        # 新增内容，用于存储scale
        self._scale_recv = []
        # 反向传播的scale
        self._g_scale_recv = []
        # 新增内容，用于存储shape
        self._shape_recv = []
        # 反向传播中的shape
        self._g_shape_recv = []
        # 新增内容，用于存储压缩后的feat
        # [N/(8/bits)*F]
        # 同理，参照feat_cpu，qt_f_cpu也应当为每一个需要通信的进程i开设一个[N/(8/bits)*F]的tensor，其中N=send_shape[i]
        self._qt_f_recv = []
        # 类比上面，反向传播的缓冲区
        self._qt_g_recv = []
        if STALE:
            self._groups = {}  # 用于存储不同epoch和layer的通信组
        if STALE:
            self.f_cnt = 0
            self.b_cnt = 0
        if STALE:
            # 为前向传播和反向传播的同步，注册信号
            self.forward_signals = {}
            self.backward_signals = {}
        # 三个参数合并的tensor，位于cpu上
        self._parm_f_cpu = []
        self._parm_g_cpu = []
        # 用于接收
        self._parm_f_cpu_recv = []
        self._parm_g_cpu_recv = []
        # 量化后数据的tensor，位于cpu上
        self._qt_f_cpu = {
            2: [],
            4: [],
            8: []
        }
        self._qt_g_cpu = {
            2: [],
            4: [],
            8: []
        }
        # 用于接收
        self._qt_f_cpu_recv = {
            2: [],
            4: [],
            8: []
        }
        self._qt_g_cpu_recv = {
            2: [],
            4: [],
            8: []
        }
        # 记录拼接后的特征
        self._concat_feat = []
        self._old_concat_feat = []
        # 记录陈旧的次数
        self.stale = []
        self.q = []
        self.bit_rank = []
        self.need_stale = []

        self.b_stale = []
        self.b_q = []
        self.b_bit_rank = []
        self.b_need_stale = []
        self._old_concat_grad = []
        self._concat_grad = []

        


    def __init_pl_pr(self):
        self._pl, self._pr = [], []
        tot = self._num_in
        for s in self._recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)


    # 这里的layer size是非线性层
    def init_buffer(self, num_in, num_all, boundary, f_recv_shape, layer_size, use_pp=False, backend='gloo',
                    pipeline=False, corr_feat=False, corr_grad=False, corr_momentum=0, bits=2):
        rank, size = dist.get_rank(), dist.get_world_size()

                 
                
        self._num_in = num_in
        self._boundary = boundary
        self._n_layers = len(layer_size)
        self._layer_size = layer_size
        self._pipeline = pipeline
        self._epoch = 0
        self._recv_shape = f_recv_shape
        self._concat_feat = [None] * self._n_layers
        self._old_concat_feat = [None] * self._n_layers
        if STALE:
            self.stale = [0] * self._n_layers
            self.q = [0] * self._n_layers
            self.bit_rank = [0] * self._n_layers
            self.need_stale = [True] * self._n_layers

            self.b_stale = [0] * self._n_layers
            self.b_q = [0] * self._n_layers
            self.b_bit_rank = [0] * self._n_layers
            self.b_need_stale = [True] * self._n_layers
            self._old_concat_grad = [None] * self._n_layers
            self._concat_grad = [None] * self._n_layers
            for i in range(self._n_layers):
                self._old_concat_grad[i] = torch.zeros(size, dtype=torch.float32, device='cuda')
                self._concat_grad[i] = torch.zeros(size, dtype=torch.float32, device='cuda')

        if STALE:
            self.f_cnt = self._n_layers - 1
            self.b_cnt = self._n_layers - 1

        if STALE:
            # 为前向传播和反向传播的同步，注册信号
            self.forward_signals = {layer: torch.tensor(0, dtype=torch.int) for layer in range(self._n_layers)}
            self.backward_signals = {layer: torch.tensor(0, dtype=torch.int) for layer in range(self._n_layers)}

        # 计算group
        # 之后可以考虑并行
        if STALE:
            for epoch in range(0, GROUP_NUM):
                for layer in range(1, self._n_layers + 1):
                    self.get_group(epoch, layer, "Forward_transfer")
                    self.get_group(epoch, layer, "Backward_transfer")
                    self.get_group(epoch, layer, "Forward_update")
                    self.get_group(epoch, layer, "Backward_update")


        if backend == 'gloo':
            self._feat_cpu, self._grad_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._f_recv_cpu, self._b_recv_cpu = [None] * self._n_layers, [None] * self._n_layers
            # 初始化用于量化和解量化的GPU缓冲区
            self._qt_f_recv = {2: [None] * self._n_layers, 4: [None] * self._n_layers, 8: [None] * self._n_layers}
            self._scale_recv = [None] * self._n_layers
            self._rmin_recv = [None] * self._n_layers
            self._shape_recv = [None] * self._n_layers            
            # 初始化反向传播中的量化和解量化的GPU缓冲区
            self._qt_g_recv = {2: [None] * self._n_layers, 4: [None] * self._n_layers, 8: [None] * self._n_layers}
            self._g_scale_recv = [None] * self._n_layers
            self._g_rmin_recv = [None] * self._n_layers
            self._g_shape_recv = [None] * self._n_layers
            # 
            self._parm_f_cpu = [None] * self._n_layers
            self._parm_f_cpu_recv = [None] * self._n_layers
            self._parm_g_cpu = [None] * self._n_layers
            self._parm_g_cpu_recv = [None] * self._n_layers
            self._qt_f_cpu = {
                2: [None] * self._n_layers,
                4: [None] * self._n_layers,
                8: [None] * self._n_layers
            }
            self._qt_f_cpu_recv = {
                2: [None] * self._n_layers,
                4: [None] * self._n_layers,
                8: [None] * self._n_layers
            }
            self._qt_g_cpu = {
                2: [None] * self._n_layers,
                4: [None] * self._n_layers,
                8: [None] * self._n_layers
            }
            self._qt_g_cpu_recv = {
                2: [None] * self._n_layers,
                4: [None] * self._n_layers,
                8: [None] * self._n_layers
            }

                            

        for i in range(self._n_layers):
            if i == 0 and use_pp:
                continue
            
            tmp1, tmp2, tmp3, tmp4 = [], [], [], []

            tmp_recv_parm_f_cpu = []
            tmp_recv_parm_g_cpu = []
            tmp_recv_qt_f_cpu = {2: [], 4: [], 8: []}
            tmp_recv_qt_g_cpu = {2: [], 4: [], 8: []}

            tmp_parm_f_cpu = []
            tmp_parm_g_cpu = []
            tmp_qt_f_cpu = {2: [], 4: [], 8: []}
            tmp_qt_g_cpu = {2: [], 4: [], 8: []}

            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                    tmp3.append(None)
                    tmp4.append(None)

                    tmp_recv_parm_f_cpu.append(None)
                    tmp_recv_parm_g_cpu.append(None)
                    tmp_recv_qt_f_cpu[2].append(None)
                    tmp_recv_qt_f_cpu[4].append(None)
                    tmp_recv_qt_f_cpu[8].append(None)
                    tmp_recv_qt_g_cpu[2].append(None)
                    tmp_recv_qt_g_cpu[4].append(None)
                    tmp_recv_qt_g_cpu[8].append(None)
                    tmp_parm_f_cpu.append(None)
                    tmp_parm_g_cpu.append(None)
                    tmp_qt_f_cpu[2].append(None)
                    tmp_qt_f_cpu[4].append(None)
                    tmp_qt_f_cpu[8].append(None)
                    tmp_qt_g_cpu[2].append(None)
                    tmp_qt_g_cpu[4].append(None)
                    tmp_qt_g_cpu[8].append(None)

                else:
                    s1 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                    s2 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1).pin_memory())
                    tmp2.append(torch.zeros(s2).pin_memory())
                    tmp3.append(torch.zeros(s2).pin_memory())
                    tmp4.append(torch.zeros(s1).pin_memory())

                    s_tmp_rmin_cpu = torch.Size([boundary[j].shape[0]])
                    s_tmp_rmin_recv_cpu = torch.Size([f_recv_shape[j]])
                    s_tmp_scale_cpu = torch.Size([boundary[j].shape[0]])
                    s_tmp_scale_recv_cpu = torch.Size([f_recv_shape[j]])
                    s_tmp_shape_cpu = torch.Size([2])
                    s_tmp_shape_recv_cpu = torch.Size([2])

                    qt_send_size_bit2 = self.calculate_packed_tensor_dimension(boundary[j].shape[0], self._layer_size[i], 2)
                    qt_recv_size_bit2 = self.calculate_packed_tensor_dimension(f_recv_shape[j], self._layer_size[i], 2)
                    qt_send_size_bit4 = self.calculate_packed_tensor_dimension(boundary[j].shape[0], self._layer_size[i], 4)
                    qt_recv_size_bit4 = self.calculate_packed_tensor_dimension(f_recv_shape[j], self._layer_size[i], 4)
                    qt_send_size_bit8 = self.calculate_packed_tensor_dimension(boundary[j].shape[0], self._layer_size[i], 8)
                    qt_recv_size_bit8 = self.calculate_packed_tensor_dimension(f_recv_shape[j], self._layer_size[i], 8)

                    s_tmp_qt_f_cpu_bit2 = torch.Size([qt_send_size_bit2])
                    s_tmp_qt_f_recv_cpu_bit2 = torch.Size([qt_recv_size_bit2])
                    s_tmp_qt_f_cpu_bit4 = torch.Size([qt_send_size_bit4])
                    s_tmp_qt_f_recv_cpu_bit4 = torch.Size([qt_recv_size_bit4])
                    s_tmp_qt_f_cpu_bit8 = torch.Size([qt_send_size_bit8])
                    s_tmp_qt_f_recv_cpu_bit8 = torch.Size([qt_recv_size_bit8])

                    tmp_recv_parm_g_cpu.append(pack_param_tensors(torch.zeros(s_tmp_rmin_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_scale_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_shape_cpu, dtype=torch.int64, pin_memory=True)))
                    tmp_recv_parm_f_cpu.append(pack_param_tensors(torch.zeros(s_tmp_rmin_recv_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_scale_recv_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_shape_recv_cpu, dtype=torch.int64, pin_memory=True)))
                    
                    tmp_recv_qt_f_cpu[2].append(torch.zeros(s_tmp_qt_f_recv_cpu_bit2, dtype=torch.int8, pin_memory=True))
                    tmp_recv_qt_f_cpu[4].append(torch.zeros(s_tmp_qt_f_recv_cpu_bit4, dtype=torch.int8, pin_memory=True))
                    tmp_recv_qt_f_cpu[8].append(torch.zeros(s_tmp_qt_f_recv_cpu_bit8, dtype=torch.int8, pin_memory=True))
                    
                    tmp_recv_qt_g_cpu[2].append(torch.zeros(s_tmp_qt_f_cpu_bit2, dtype=torch.int8, pin_memory=True))
                    tmp_recv_qt_g_cpu[4].append(torch.zeros(s_tmp_qt_f_cpu_bit4, dtype=torch.int8, pin_memory=True))
                    tmp_recv_qt_g_cpu[8].append(torch.zeros(s_tmp_qt_f_cpu_bit8, dtype=torch.int8, pin_memory=True))
                    
                    tmp_parm_g_cpu.append(pack_param_tensors(torch.zeros(s_tmp_rmin_recv_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_scale_recv_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_shape_recv_cpu, dtype=torch.int64, pin_memory=True)))
                    tmp_parm_f_cpu.append(pack_param_tensors(torch.zeros(s_tmp_rmin_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_scale_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_shape_cpu, dtype=torch.int64, pin_memory=True)))
                    
                    tmp_qt_f_cpu[2].append(torch.zeros(s_tmp_qt_f_cpu_bit2, dtype=torch.int8, pin_memory=True))
                    tmp_qt_f_cpu[4].append(torch.zeros(s_tmp_qt_f_cpu_bit4, dtype=torch.int8, pin_memory=True))
                    tmp_qt_f_cpu[8].append(torch.zeros(s_tmp_qt_f_cpu_bit8, dtype=torch.int8, pin_memory=True))
                    
                    tmp_qt_g_cpu[2].append(torch.zeros(s_tmp_qt_f_recv_cpu_bit2, dtype=torch.int8, pin_memory=True))
                    tmp_qt_g_cpu[4].append(torch.zeros(s_tmp_qt_f_recv_cpu_bit4, dtype=torch.int8, pin_memory=True))
                    tmp_qt_g_cpu[8].append(torch.zeros(s_tmp_qt_f_recv_cpu_bit8, dtype=torch.int8, pin_memory=True))

            self._feat_cpu[i] = tmp1
            self._f_recv_cpu[i] = tmp3

            if i > 0:
                self._grad_cpu[i] = tmp2
                self._b_recv_cpu[i] = tmp4

            self._parm_f_cpu_recv[i] = tmp_recv_parm_f_cpu
            self._qt_f_cpu_recv[2][i] = tmp_recv_qt_f_cpu[2]
            self._qt_f_cpu_recv[4][i] = tmp_recv_qt_f_cpu[4]
            self._qt_f_cpu_recv[8][i] = tmp_recv_qt_f_cpu[8]
            if i > 0:
                self._parm_g_cpu_recv[i] = tmp_recv_parm_g_cpu
                self._qt_g_cpu_recv[2][i] = tmp_recv_qt_g_cpu[2]
                self._qt_g_cpu_recv[4][i] = tmp_recv_qt_g_cpu[4]
                self._qt_g_cpu_recv[8][i] = tmp_recv_qt_g_cpu[8]

            self._parm_f_cpu[i] = tmp_parm_f_cpu
            self._qt_f_cpu[2][i] = tmp_qt_f_cpu[2]
            self._qt_f_cpu[4][i] = tmp_qt_f_cpu[4]
            self._qt_f_cpu[8][i] = tmp_qt_f_cpu[8]
            if i > 0:
                self._parm_g_cpu[i] = tmp_parm_g_cpu
                self._qt_g_cpu[2][i] = tmp_qt_g_cpu[2]
                self._qt_g_cpu[4][i] = tmp_qt_g_cpu[4]
                self._qt_g_cpu[8][i] = tmp_qt_g_cpu[8]





        self._f_buf = [None] * self._n_layers
        self._comm_stream = torch.cuda.Stream()
        self._backend = backend
        self._f_recv, self._b_recv = [None] * self._n_layers, [None] * self._n_layers
        self._f_cpu_event, self._b_cpu_event = [None] * self._n_layers, [None] * self._n_layers
        self._f_cuda_event, self._b_cuda_event = [None] * self._n_layers, [None] * self._n_layers



        for i in range(self._n_layers):
            if i == 0 and use_pp:
                continue
            self._f_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda')
            tmp1, tmp2 = [], []
            tmp_qt_f_recv = {2: [], 4: [], 8: []}
            tmp_scale_recv, tmp_rmin_recv, tmp_shape_recv = [], [], []
            tmp_qt_g_recv = {2: [], 4: [], 8: []}
            tmp_g_scale_recv, tmp_g_rmin_recv, tmp_g_shape_recv = [], [], []

            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)

                    tmp_qt_f_recv[2].append(None)
                    tmp_qt_f_recv[4].append(None)
                    tmp_qt_f_recv[8].append(None)
                    tmp_scale_recv.append(None)
                    tmp_rmin_recv.append(None)
                    tmp_shape_recv.append(None)

                    tmp_qt_g_recv[2].append(None)
                    tmp_qt_g_recv[4].append(None)
                    tmp_qt_g_recv[8].append(None)
                    tmp_g_scale_recv.append(None)
                    tmp_g_rmin_recv.append(None)
                    tmp_g_shape_recv.append(None)

                else:
                    s1 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda'))
                    tmp2.append(torch.zeros(s2, device='cuda'))

                    s_tmp_rmin_recv = torch.Size([f_recv_shape[j]])
                    s_tmp_scale_recv = torch.Size([f_recv_shape[j]])
                    s_tmp_shape_recv = torch.Size([2])
                    qt_recv_size_bit2 = self.calculate_packed_tensor_dimension(f_recv_shape[j], self._layer_size[i], 2)
                    qt_recv_size_bit4 = self.calculate_packed_tensor_dimension(f_recv_shape[j], self._layer_size[i], 4)
                    qt_recv_size_bit8 = self.calculate_packed_tensor_dimension(f_recv_shape[j], self._layer_size[i], 8)

                    s_tmp_rmin_gpu = torch.Size([boundary[j].shape[0]])
                    s_tmp_scale_gpu = torch.Size([boundary[j].shape[0]])
                    s_tmp_shape_gpu = torch.Size([2])
                    qt_send_size_bit2 = self.calculate_packed_tensor_dimension(boundary[j].shape[0], self._layer_size[i], 2)
                    qt_send_size_bit4 = self.calculate_packed_tensor_dimension(boundary[j].shape[0], self._layer_size[i], 4)
                    qt_send_size_bit8 = self.calculate_packed_tensor_dimension(boundary[j].shape[0], self._layer_size[i], 8)

                    s_tmp_qt_f_recv_bit2 = torch.Size([qt_recv_size_bit2])
                    s_tmp_qt_f_gpu_bit2 = torch.Size([qt_send_size_bit2])
                    s_tmp_qt_f_recv_bit4 = torch.Size([qt_recv_size_bit4])
                    s_tmp_qt_f_gpu_bit4 = torch.Size([qt_send_size_bit4])
                    s_tmp_qt_f_recv_bit8 = torch.Size([qt_recv_size_bit8])
                    s_tmp_qt_f_gpu_bit8 = torch.Size([qt_send_size_bit8])

                    tmp_qt_f_recv[2].append(torch.zeros(s_tmp_qt_f_recv_bit2, dtype=torch.int8, device='cuda'))
                    tmp_qt_f_recv[4].append(torch.zeros(s_tmp_qt_f_recv_bit4, dtype=torch.int8, device='cuda'))
                    tmp_qt_f_recv[8].append(torch.zeros(s_tmp_qt_f_recv_bit8, dtype=torch.int8, device='cuda'))
                    tmp_scale_recv.append(torch.zeros(s_tmp_scale_recv, dtype=torch.bfloat16, device='cuda'))
                    tmp_rmin_recv.append(torch.zeros(s_tmp_rmin_recv, dtype=torch.bfloat16, device='cuda'))
                    tmp_shape_recv.append(torch.zeros(s_tmp_shape_recv, dtype=torch.int64, device='cuda'))

                    tmp_qt_g_recv[2].append(torch.zeros(s_tmp_qt_f_gpu_bit2, dtype=torch.int8, device='cuda'))
                    tmp_qt_g_recv[4].append(torch.zeros(s_tmp_qt_f_gpu_bit4, dtype=torch.int8, device='cuda'))
                    tmp_qt_g_recv[8].append(torch.zeros(s_tmp_qt_f_gpu_bit8, dtype=torch.int8, device='cuda'))
                    tmp_g_scale_recv.append(torch.zeros(s_tmp_scale_gpu, dtype=torch.bfloat16, device='cuda'))
                    tmp_g_rmin_recv.append(torch.zeros(s_tmp_rmin_gpu, dtype=torch.bfloat16, device='cuda'))
                    tmp_g_shape_recv.append(torch.zeros(s_tmp_shape_gpu, dtype=torch.int64, device='cuda'))

            self._f_recv[i] = tmp1
            if i > 0:
                self._b_recv[i] = tmp2

            # 接收缓冲区初始化
            self._qt_f_recv[2][i] = tmp_qt_f_recv[2]
            self._qt_f_recv[4][i] = tmp_qt_f_recv[4]
            self._qt_f_recv[8][i] = tmp_qt_f_recv[8]
            self._scale_recv[i] = tmp_scale_recv
            self._rmin_recv[i] = tmp_rmin_recv
            self._shape_recv[i] = tmp_shape_recv

            # 反向传播中的接收缓冲区初始化
            if i > 0:
                self._qt_g_recv[2][i] = tmp_qt_g_recv[2]
                self._qt_g_recv[4][i] = tmp_qt_g_recv[4]
                self._qt_g_recv[8][i] = tmp_qt_g_recv[8]
                self._g_scale_recv[i] = tmp_g_scale_recv
                self._g_rmin_recv[i] = tmp_g_rmin_recv
                self._g_shape_recv[i] = tmp_g_shape_recv

            self._f_cpu_event[i] = Event()
            self._b_cpu_event[i] = Event()
            self._f_cuda_event[i] = torch.cuda.Event()
            self._b_cuda_event[i] = torch.cuda.Event()



        self._pool = ThreadPool(processes=2 * self._n_layers)
        self.__init_pl_pr()


    def __qt_forward_gloo_all_to_all(self, feat, layer, tag, bits, epoch, forward=True):
        with true_comm_timer.timer(f'forward_{epoch}_{layer}'):
            with forward_comm_timer.timer(f'_{epoch}_{layer}'):
                rank, size = dist.get_rank(), dist.get_world_size()
                req_send, req_recv = [], queue.Queue()
                self._comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._comm_stream):
                    if QT_ON_COMP == False:
                        self.quantization_to_cpu(feat, self._qt_f_cpu[bits][layer], self._parm_f_cpu[layer], bits, True, epoch, layer)

                    for i in range(1, size):
                        left = (rank - i + size) % size
                        right = (rank + i) % size

                        r1_0 = dist.isend(self._qt_f_cpu[bits][layer][right], tag=tag, dst=right)
                        req_send.append(r1_0)
                        r1_1 = dist.isend(self._parm_f_cpu[layer][right], tag=tag + self._n_layers, dst=right)
                        req_send.append(r1_1)

                        r2_0 = dist.irecv(self._qt_f_cpu_recv[bits][layer][left], tag=tag, src=left)
                        r2_1 = dist.irecv(self._parm_f_cpu_recv[layer][left], tag=tag + self._n_layers, src=left)
                        req_recv.put((r2_0, r2_1, left))

                    while not req_recv.empty():
                        r_0, r_1, idx = req_recv.get()
                        r_0.wait()
                        r_1.wait()

                        self.dequantization(idx, self._f_recv[layer], self._parm_f_cpu_recv[layer], self._qt_f_cpu_recv[bits][layer], self._qt_f_recv[bits][layer], self._scale_recv[layer], self._rmin_recv[layer], self._shape_recv[layer], bits, forward, epoch, layer)

                    for r in req_send:
                        r.wait()

    def __qt_backward_gloo_all_to_all(self, grad, layer, tag, bits, epoch, forward=False):
        with true_comm_timer.timer(f'backward{epoch}_{layer}'):
            with backward_comm_timer.timer(f'_{epoch}_{layer}'):
                rank, size = dist.get_rank(), dist.get_world_size()
                req_send, req_recv = [], queue.Queue()
                self._comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._comm_stream):
                    if QT_ON_COMP == False:
                        self.quantization_to_cpu(grad, self._qt_g_cpu[bits][layer], self._parm_g_cpu[layer], bits, False, epoch, layer)
                    for i in range(1, size):
                        left = (rank - i + size) % size
                        right = (rank + i) % size

                        r1_0 = dist.isend(self._qt_g_cpu[bits][layer][right], tag=tag, dst=right)
                        req_send.append(r1_0)
                        r1_1 = dist.isend(self._parm_g_cpu[layer][right], tag=tag + self._n_layers, dst=right)
                        req_send.append(r1_1)

                        r2_0 = dist.irecv(self._qt_g_cpu_recv[bits][layer][left], tag=tag, src=left)
                        r2_1 = dist.irecv(self._parm_g_cpu_recv[layer][left], tag=tag + self._n_layers, src=left)
                        req_recv.put((r2_0, r2_1, left))

                    while not req_recv.empty():
                        r_0, r_1, idx = req_recv.get()
                        r_0.wait()
                        r_1.wait()
                        self.dequantization(idx, self._b_recv[layer], self._parm_g_cpu_recv[layer], self._qt_g_cpu_recv[bits][layer], self._qt_g_recv[bits][layer], self._g_scale_recv[layer], self._g_rmin_recv[layer], self._g_shape_recv[layer], bits, forward, epoch, layer)

                    for r in req_send:
                        r.wait()



# 前向传播，不使用量化
    def __forward_gloo_all_to_all(self, feat, layer, tag, epoch, forward=True):
        with true_comm_timer.timer(f'forward_{epoch}_{layer}'):
            with forward_comm_timer.timer(f'_{epoch}_{layer}'):
                rank, size = dist.get_rank(), dist.get_world_size()
                req_send, req_recv = [], queue.Queue()

                self._comm_stream.wait_stream(torch.cuda.current_stream()) 
                with torch.cuda.stream(self._comm_stream):
                    for i in range(1, size):
                        left = (rank - i + size) % size
                        right = (rank + i) % size
                        self._feat_cpu[layer][right].copy_(feat[self._boundary[right]])

                        r1 = dist.isend(self._feat_cpu[layer][right], tag=tag, dst=right)
                        req_send.append(r1)
                        r2 = dist.irecv(self._f_recv_cpu[layer][left], tag=tag, src=left)
                        req_recv.put((r2, left))  

                    while not req_recv.empty():
                        r, idx = req_recv.get()
                        r.wait()
                        self._f_recv[layer][idx].copy_(self._f_recv_cpu[layer][idx], non_blocking=True)

                    for r in req_send:
                        r.wait()



# 反向传播, 不使用量化
    def __backward_gloo_all_to_all(self, grad, layer, tag, epoch, forward=False):
        with true_comm_timer.timer(f"backward_{epoch}__{layer}"):
            with backward_comm_timer.timer(f"_{epoch}_{layer}"):
                rank, size = dist.get_rank(), dist.get_world_size()
                req_send, req_recv = [], queue.Queue()

                self._comm_stream.wait_stream(torch.cuda.current_stream()) 
                with torch.cuda.stream(self._comm_stream):
                    for i in range(1, size):
                        left = (rank - i + size) % size
                        right = (rank + i) % size
                        # 接收不能用non_blocking
                        self._grad_cpu[layer][right].copy_(grad[self._pl[right]:self._pr[right]])

                        r1 = dist.isend(self._grad_cpu[layer][right], tag=tag, dst=right)
                        req_send.append(r1)

                        r2 = dist.irecv(self._b_recv_cpu[layer][left], tag=tag, src=left)
                        req_recv.put((r2, left))  


                    while not req_recv.empty():
                        
                        r, idx = req_recv.get()
                        r.wait()
                        self._b_recv[layer][idx].copy_(self._b_recv_cpu[layer][idx], non_blocking = True)

                    for r in req_send:
                        r.wait()     



    def next_epoch(self):
        self._epoch += 1


    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = []
        for i in range(size):
            if i != rank:
                tmp.append(self._f_recv[layer][i])
                # print(f"self._f_recv[{layer}][{i}]: {self._f_recv[layer][i].shape}")
        # print(f"feat: {feat.shape}")

        new_concat_feat = torch.cat(tmp) if tmp else torch.tensor([])

        if self._concat_feat[layer] is not None:
            # 保存旧的 concat_feat
            self._old_concat_feat[layer] = self._concat_feat[layer]
            # 将新的特征拼接到旧的 concat_feat 上
            self._concat_feat[layer] = new_concat_feat
        else:
            self._concat_feat[layer] = new_concat_feat

        return torch.cat([feat, self._concat_feat[layer]])

    def get_group(self, epoch, layer, _type):
        key = (epoch % GROUP_NUM, layer, _type)
        if key not in self._groups:
            # dist.barrier()
            if TRANFER_DEBUG:
                print(f"Creating new group for epoch: {epoch}, layer: {layer}, type: {_type}")
            ranks = list(range(dist.get_world_size()))  
            self._groups[key] = dist.new_group(ranks=ranks)
        else:
            if TRANFER_DEBUG:
                print(f"Using existing group for epoch: {epoch}, layer: {layer}")
        return self._groups[key]


    
    def quantization_to_cpu(self, feat, qt_cpu, parm_cpu, bits, forward, epoch, layer):
        rank, size = dist.get_rank(), dist.get_world_size()

        for i in range(1, size):
            right = (rank + i) % size
            if forward:
                with qt_timer.timer(f"forward_pack_{epoch}_{layer}_{i}"):
                    qt, scale, rmin, shape = message_quantization(feat[self._boundary[right]], bits, stochastic=True)
            else:
                with qt_timer.timer(f"backward_pack_{epoch}_{layer}_{i}"):
                    qt, scale, rmin, shape = message_quantization(feat[self._pl[right]:self._pr[right]], bits, stochastic=True)

            shape_tensor = torch.tensor(shape)
            with pack_timer.timer(f"{forward}_pack_{epoch}_{layer}_{i}"):
                parm = pack_param_tensors(rmin, scale, shape_tensor)
                # qt_cpu[right].copy_(qt, non_blocking = True)
                # parm_cpu[right].copy_(parm, non_blocking = True)
            qt_cpu[right] = qt.to('cpu')
            parm_cpu[right] = parm.to('cpu')


 
    # 解量化
    def dequantization(self, idx, dst, parm_cpu, qt_cpu, qt_gpu, scale_gpu, rmin_gpu, shape_gpu, bits, forward, epoch, layer):
        rank, size = dist.get_rank(), dist.get_world_size()

        with unpack_timer.timer(f"{forward}_pack_{epoch}_{layer}_{idx}"):
            rmin_cpu, scale_cpu, shape_cpu = unpack_param_tensors(parm_cpu[idx])
        
        qt_gpu[idx].copy_(qt_cpu[idx])
        rmin_gpu[idx].copy_(rmin_cpu)
        scale_gpu[idx].copy_(scale_cpu)
        shape_gpu[idx].copy_(shape_cpu)
        shape = (shape_gpu[idx][0].item(), shape_gpu[idx][1].item())
            # shape = (shape_cpu[0].item(), shape_cpu[1].item())
            # 可以改成copy？
        with dq_timer.timer(f"{forward}_pack_{epoch}_{layer}_{idx}"):
            dst[idx] = message_dequantization(qt_gpu[idx], scale_gpu[idx], rmin_gpu[idx], shape, bits)
    



    def inverted_cosine_similarity(self, old_data, new_data):
        # 获取当前层的两个特征向量
        current_feat = new_data
        old_feat = old_data
        
        # 将两个张量展平成一维向量
        current_feat_flat = current_feat.view(-1)
        old_feat_flat = old_feat.view(-1)
        
        # 计算两个张量之间的余弦相似度
        cos_sim = F.cosine_similarity(current_feat_flat, old_feat_flat, dim=0)
        
        # 将1映射到0，将-1映射到1，即进行反转
        inverted_cos_sim = (1 - cos_sim) / 2
        
        return inverted_cos_sim

    def bit_assigner(self, epoch, layer, old_data, new_data, stale_cnt, q_cnt, need_stale, bit_rank, is_forward=True):
        bit_options = BIT_OPTIONS
        bit_max_rank = len(bit_options) - 1
        # Calculate tensor_diff_sum as the sum of the absolute differences
        if old_data[layer] is None or epoch == 0:
            return bit_options[bit_rank[layer]]
        else:
            # tensor_diff_sum = torch.sum(torch.abs(self._concat_feat[layer] - self._old_concat_feat[layer]))
            tensor_diff_sum = self.inverted_cosine_similarity(old_data[layer], new_data[layer])
            # print(f"layer: {layer} diff: {tensor_diff_sum}")
        
        mapped_value = tensor_diff_sum
        # # Apply a smooth mapping function
        # mapped_value = 1 - torch.exp(-0.5 * tensor_diff_sum)
        # print(f"map value:{mapped_value}")
        # # Ensure the value is in the range [0, 1]
        # mapped_value = torch.clamp(mapped_value, 0, 1)
        # print(f"map value smooth:{mapped_value}")
        if is_forward:
            print(f"forward layer {layer} diff: {mapped_value}")
        else:
            print(f"backward layer {layer} diff: {mapped_value}")
        if mapped_value > E:
            # if self.stale[layer] > 1.5:
            #     return -2  
            need_stale[layer] = False
            print(f"layer {layer}:set false")

            if stale_cnt[layer] > 0:
                q_cnt[layer] = stale_cnt[layer]
                stale_cnt[layer] = 0
                bit_rank[layer] = 0

            else:
                result = 1 / (1 + torch.exp(-K * tensor_diff_sum / q_cnt[layer]))
                # print(f"layer: {layer} result: {result}")
                if result > 0.5:
                    bit_rank[layer] = min(bit_rank[layer] + 1, bit_max_rank)
                else:
                    bit_rank[layer] = max(bit_rank[layer] - 1, 0)
                q_cnt[layer] += 1

        else:
            need_stale[layer] = True

        # print(f"layer: {layer} bit: {bit_options[self.bit_rank[layer]]}")
        return bit_options[bit_rank[layer]]


    # 自适应比特分配
    def adaptive_bit_allocation(self, epoch, layer, bits, update_group, rank, is_forward=True):
        start_time = time.time()
        # if is_forward:
        #     if self._old_concat_feat[layer] is None:
        #         tensor_diff_sum = 0
        #     else:
        #         tensor_diff_sum = torch.sum((self._concat_feat[layer] - self._old_concat_feat[layer]) / bits)
        #         # print(f"sum: {tensor_diff_sum}")

        signals = self.forward_signals if is_forward else self.backward_signals
        event = self._f_cpu_event if is_forward else self._b_cpu_event
        old_data = self._old_concat_feat if is_forward else self._old_concat_grad
        new_data = self._concat_feat if is_forward else self._concat_grad

        stale_cnt = self.stale if is_forward else self.b_stale
        q_cnt = self.q if is_forward else self.b_q
        bit_rank = self.bit_rank if is_forward else self.b_bit_rank
        need_stale = self.need_stale if is_forward else self.b_need_stale


        # k = (epoch * 3 + layer - 1) / (self.f_cnt if is_forward else self.b_cnt)
        # if k > STALE_THRESHOLD:
        #     return 1, bits
        # k = (epoch * 2 + layer - 1) / (self.f_cnt if is_forward else self.b_cnt)
        # Rank 0 进程进行判断
        if rank == 0:
            condition_met = event[layer].is_set()
            # 上一个通讯结束
            if condition_met:
                if STALE and ADAPTIVE:
                    # 返回bit
                    selected_bits = self.bit_assigner(epoch, layer, old_data, new_data, stale_cnt, q_cnt, need_stale, bit_rank, is_forward)
                else:
                    selected_bits = bits  # 使用传入的bits值
                signals[layer] = torch.tensor(selected_bits, dtype=torch.int)
            # 上一个通讯未结束
            elif STALE and ADAPTIVE and need_stale[layer]  == False:
                signals[layer] = torch.tensor(bits, dtype=torch.int)    
                # print(f"layer {layer} wait bit: {signals[layer]}")  
            # elif STALE and (not ADAPTIVE) and k > STALE_THRESHOLD:
            #     signals[layer] = torch.tensor(bits, dtype=torch.int) 

            else:
                selected_bits = bits  # 保持原位宽
                stale_cnt[layer] += 1
                signals[layer] = torch.tensor(-1, dtype=torch.int)

            # 将条件信号和比特数编码到同一个tensor中
            # 应该是这里太费时了
        # 广播信号给所有进程
        dist.broadcast(signals[layer], src=0, group=update_group)

        # 解码比特数
        # 通讯未结束，跳过
        if signals[layer].item() == -1:
            condition_met = 0
            selected_bits = bits
        else:
            # 进行通讯
            condition_met = 1
            selected_bits = signals[layer].item()

        end_time = time.time()
        # if rank == 0:
        #     print(end_time - start_time)
        return condition_met, selected_bits



    # 消息传递
    # 每层都会进行一次update
    # 同步部分执行的很快。主要开销在wait部分，wait也保证了每次return回来的f_buf是更新过的。如果采用skip的方式，我认为只需要在有通讯时返回f_buf即可
    def update(self, layer, feat, bits, epoch, rank):
        # 位置：在关键点（如数据传输前后）确保所有CUDA操作完成。
        # 作用：同步当前默认流，确保当前流中的所有操作都已完成。
        # 目的：阻塞CPU端的执行，直到当前默认流中的所有CUDA操作都完成。通常用于确保在执行后续操作前，所有之前的CUDA操作都已完成。
        if WAIT_REDUCE:
            torch.cuda.current_stream().synchronize()
        # 这里可能得加一个计算时间
        # 有在初始化缓冲区时提前计算，如果没有，会自己算，并且计入其他时间
        if FORWARD_QUANTIZATION:
            selected_bits = bits
        else:
            selected_bits = 0
        if STALE:
            f_transfer_group = self.get_group(epoch, layer, "Forward_transfer")
            b_transfer_group = self.get_group(epoch, layer, "Backward_transfer")
            f_update_group = self.get_group(epoch, layer, "Forward_update")
            b_update_group = self.get_group(epoch, layer, "Backward_update")
        else:
            f_transfer_group = False
            b_transfer_group = False
            f_update_group = False
            b_update_group = False




        if self._pipeline is False:
            with comm_timer.timer(f'forward_{layer}_{epoch}'):
                if FORWARD_QUANTIZATION and QT_ON_COMP and selected_bits != 0:
                    self.quantization_to_cpu(feat, self._qt_f_cpu[layer], self._parm_f_cpu[layer], bits, True, epoch, layer)
                self.__feat_transfer(epoch, layer, feat, bits, f_transfer_group)
                # 位置：在依赖通信流操作的后续计算操作之前。
                # 作用：让当前默认流等待事件 self._f_cuda_event[layer] 的完成。
                # 目的：确保当前默认流中的操作在 self._f_cuda_event[layer] 事件触发后才开始执行，即等待通信流中的特定操作完成。
                torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])

            self._f_buf[layer] = self.__feat_concat(layer, feat)
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(epoch, layer, bits, b_transfer_group, b_update_group, rank))
            return self._f_buf[layer]
        else:
            # 如果不是第一个epoch，等待上一个epoch的CPU和CUDA事件完成，并清除CPU事件
            if epoch > 0:
                # 统计的实际上是等待上一个epoch相同层结束的时间
                with comm_timer.timer(f'forward_{layer}'):
                    # 确保前置操作完成，协调CPU和GPU操作，准备下一次操作
                    # self._f_cpu_event[layer].wait() 保证了在继续执行后续操作之前，某些前置操作已经在其他线程或进程中完成。
                    # 比如，这可能是一些数据准备或计算操作
                    # torch.cuda.current_stream().wait_event(self._f_cuda_event[layer]) 使得当前CUDA流中的操作
                    # 在 self._comm_stream 流中的操作完成之后才开始。这样做确保了数据的一致性和正确性，避免了数据竞争和潜在的错误
                    # self._f_cpu_event[layer].clear() 清除事件状态，为下一次等待操作做好准备。
                    # 这意味着可以重复使用同一个事件对象进行多次同步操作
                    # 在feat_transfer结束时，有_f_cpu_event[layer].set()
                    # 等待上一个epoch的该层通讯完成
                    if STALE:
                        # 计算陈旧阈值
                        
                        condition_met, selected_bits = self.adaptive_bit_allocation(epoch, layer, BIT_OPTIONS[self.bit_rank[layer]], f_update_group, rank, is_forward=True)                   
                        # dist.barrier(f_update_group)
                        if condition_met == 1:
                            if Count_Stale:
                                print(f"forward transfer {epoch} {layer}")

                            self.f_cnt += 1
                            self._f_cpu_event[layer].wait() 
                            torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
                            self._f_cpu_event[layer].clear()
                        else:
                            # 这一步有问题，可能会导致写冲突
                            # self._f_buf[layer] = self.__feat_concat(layer, feat)
                            # tmp_f_buf = self.__feat_concat(layer, feat)
                            return torch.cat([feat, self._concat_feat[layer]])

                    else:
                        self._f_cpu_event[layer].wait()
                        torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
                        self._f_cpu_event[layer].clear()
            # BNS-GCN的这一步在传输后面，可见是先计算再传播
            # 第二次到这的时候卡了，应该是
            self._f_buf[layer] = self.__feat_concat(layer, feat)
            # 量化
            if FORWARD_QUANTIZATION and QT_ON_COMP and selected_bits != 0:
                self.quantization_to_cpu(feat, self._qt_f_cpu[layer], self._parm_f_cpu[layer], selected_bits, True, epoch, layer)
                    
            # 使用异步线程池 _pool 进行异步特征传输，以提高效率
            if rank == 0:
                print(f"forward layer:{layer}, bit:{selected_bits}")
            self._pool.apply_async(self.__feat_transfer, args=(epoch, layer, feat, selected_bits, f_transfer_group))
            # 通过注册梯度钩子 __grad_hook，确保梯度在反向传播过程中正确传递和更新
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(epoch, layer, selected_bits, b_transfer_group, b_update_group, rank))
            # return True, self._f_buf[layer]
            return self._f_buf[layer]


    def get_feat(self, feat, layer):
        # return torch.cat([feat, self._concat_feat[layer]])
        return self.__feat_concat(layer, feat)



    def __feat_transfer(self, epoch, layer, feat, bits, group):
        tag = epoch * 4 * self._n_layers + layer

        if TRANFER_DEBUG:
            print(f"Starting feature transfer for tag: {tag}")

        # if STALE:
        #     print("go to get group")
        #     group = self.get_group(epoch, layer)
        #     if TRANFER_DEBUG:
        #         print(f"Using stale group: {group}")

        if self._backend == 'gloo':
            if TRANFER_DEBUG:
                print(f"Using backend: gloo")
            
            # 在此处选择，通讯是否加量化
            if FORWARD_QUANTIZATION:
                if TRANFER_DEBUG:
                    print(f"Forward quantization without stale check for layer: {layer}")
                if bits == 0:
                    self.__forward_gloo_all_to_all(feat, layer, tag, epoch)
                else:
                    self.__qt_forward_gloo_all_to_all(feat, layer, tag, bits, epoch, True)
            else:
                if TRANFER_DEBUG:
                    print(f"Standard all-to-all communication for layer: {layer}")
                self.__forward_gloo_all_to_all(feat, layer, tag, epoch)

            self._f_cuda_event[layer].record(self._comm_stream)

            if TRANFER_DEBUG:
                print(f"CUDA event recorded for layer: {layer}")
        else:
            raise NotImplementedError
        if TRANFER_DEBUG:
            print(f"Feature transfer completed for tag: {tag}")

        if STALE: 
            dist.barrier(group)
            if TRANFER_DEBUG:
                print(f"Barrier for group: {group}")
            if Count_Time:
                set_time = time.time()
                print(f"forward set event begin*** epoch: {epoch} layer: {layer} start time: {set_time}")
                
                dist.barrier(group)

        self._f_cpu_event[layer].set()
        if TRANFER_DEBUG:
            print(f"CPU event set for layer: {layer}")

    def __update_grad(self, epoch, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        
        # 确保self._old_concat_grad和self._new_concat_grad是在CUDA上
        self._old_concat_grad[layer] = self._concat_grad[layer]
        if epoch == 0:
            self._old_concat_grad[layer] = torch.zeros(grad.shape, dtype=torch.float32, device=grad.device)
        # 创建一个在CUDA上的临时张量
        tmp = torch.zeros(grad.shape, dtype=torch.float32, device=grad.device)
        
        for i in range(size):
            if i == rank:
                continue
            else:
                # 打印相关张量的维度
                # print(f"self._boundary[i].shape: {self._boundary[i].shape}")
                # print(f"self._b_recv[layer][i].shape: {self._b_recv[layer][i].shape}")
                # print(f"tmp.shape: {tmp.shape}")
                # print(f"grad.shape: {grad.shape}")
                tmp[self._boundary[i]] += self._b_recv[layer][i]
        
        grad += tmp
        self._concat_grad[layer] = tmp



    # 类比Update，在update中被注册，在Loss.backward中被调用
    # 应当不需要进行修改，只需要修改grad_transfer即可
    def __grad_hook(self, epoch, layer, bits, transfer_group, update_group, rank):
        def fn(grad):
            if WAIT_REDUCE:
                torch.cuda.current_stream().synchronize()
            if BACKWARD_QUANTIZATION:
                selected_bits = bits
            else:
                selected_bits = 0
            if self._pipeline is False:
                with comm_timer.timer(f'backward_{layer}'):
                    if BACKWARD_QUANTIZATION and QT_ON_COMP and selected_bits != 0:
                        self.quantization_to_cpu(grad, self._qt_g_cpu[layer], self._parm_g_cpu[layer], bits, False, epoch, layer)
                    self.__grad_transfer(epoch, layer, grad, bits, transfer_group)
                    torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                self.__update_grad(epoch, layer, grad)
                return grad
            else:
                if self._epoch > 0:
                    with comm_timer.timer(f'backward_{layer}'):
                        if STALE:
                            condition_met, selected_bits = self.adaptive_bit_allocation(epoch, layer, bits, update_group, rank, is_forward=False)                   
                            if condition_met == 0:
                                grad += self._concat_grad[layer]
                                # self.__update_grad(layer, grad)
                                return grad
                            else:
                                if Count_Stale:
                                    print(f"forward transfer {epoch} {layer}")
                                self.b_cnt += 1
                                self._b_cpu_event[layer].wait()    
                                torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                                self._b_cpu_event[layer].clear()

                        else:
                            
                            self._b_cpu_event[layer].wait()
                            torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                            self._b_cpu_event[layer].clear()

                self.__update_grad(epoch, layer, grad)
                if BACKWARD_QUANTIZATION and QT_ON_COMP and selected_bits != 0:
                    self.quantization_to_cpu(grad, self._qt_g_cpu[layer], self._parm_g_cpu[layer], selected_bits, False, epoch, layer)
                if rank == 0:
                    print(f"backward layer:{layer}, bit:{selected_bits}")       
                self._pool.apply_async(self.__grad_transfer, args=(epoch, layer, grad, selected_bits, transfer_group))
                return grad
        return fn

    # 类比feat_transfer
    def __grad_transfer(self, epoch, layer, grad, bits, group):
        tag = epoch * 4 * self._n_layers + layer + self._n_layers * 2

        if self._backend == 'gloo':
            if BACKWARD_QUANTIZATION:
                if bits == 0:
                    self.__backward_gloo_all_to_all(grad, layer, tag, epoch)
                else:
                    self.__qt_backward_gloo_all_to_all(grad, layer, tag, bits, epoch, False)
            else:
                self.__backward_gloo_all_to_all(grad, layer, tag, epoch)

            self._b_cuda_event[layer].record(self._comm_stream)
        else:
            raise NotImplementedError
        
        if STALE:
            dist.barrier(group)
            if TRANFER_DEBUG:
                print(f"Backward Barrier for group: {group}")
            if Count_Time:
                set_time = time.time()
                print(f"backward set event*** epoch: {epoch} layer: {layer} start time: {set_time}")
                dist.barrier(group)

        self._b_cpu_event[layer].set()

