import torch
from multiprocessing.pool import ThreadPool
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
        # 注意
        self._f_avg, self._b_avg = [], []
        self._recv_shape = []
        # 线程池，可以多加了解
        self._pool = None
        # 通讯流和校正流
        self._comm_stream, self._corr_stream = None, None
        # 特征和梯度的cpu、cuda事件
        # 注意，如果要增加量化，是否需要多注册事件
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []
        self._backend = None
        # 校正流，用于引入修正以提高收敛性
        self._corr_momentum = 0
        self._corr_feat, self._corr_grad = False, False
        self._pl, self._pr = [], []
        # 新增内容
        # 新增内容，用于存储min
        self._rmin_gpu = []
        self._rmin_recv = []
        # 反向传播中的rmin
        self._g_rmin_gpu = []
        self._g_rmin_recv = []
        # 新增内容，用于存储scale
        self._scale_gpu = []
        self._scale_recv = []
        # 反向传播的scale
        self._g_scale_gpu = []
        self._g_scale_recv = []
        # 新增内容，用于存储shape
        self._shape_gpu = []
        self._shape_recv = []
        # 反向传播中的shape
        self._g_shape_gpu = []
        self._g_shape_recv = []
        # 新增内容，用于存储压缩后的feat
        # [N/(8/bits)*F]
        # 同理，参照feat_cpu，qt_f_cpu也应当为每一个需要通信的进程i开设一个[N/(8/bits)*F]的tensor，其中N=send_shape[i]
        self._qt_f_gpu = []
        self._qt_f_recv = []
        # 类比上面，反向传播的缓冲区
        self._qt_g_gpu = []
        self._qt_g_recv = []
        # 新增内容，用于存储合并后的tensor
        self.recv_cpu_tensor = []
        self.send_cpu_tensor = []
        # 反向传播中的tensor
        self.g_send_cpu_tensor = []
        self.g_recv_cpu_tensor = []
        if STALE_CHECK_2:
            # 根据陈旧，判断是否进行发送
            self.active = True
            self.ver = [] * self._n_layers
            self.ver_back = [] * self._n_layers
        if STALE:
            self._groups = {}  # 用于存储不同epoch和layer的通信组
        if STALE:
            self.f_cnt = 3
            self.b_cnt = 3
        if STALE:
            # 为前向传播和反向传播的同步，注册信号
            self.forward_signals = {layer: torch.tensor(0, dtype=torch.int) for layer in range(4)}
            self.backward_signals = {layer: torch.tensor(0, dtype=torch.int) for layer in range(4)}


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
            self._qt_f_gpu, self._scale_gpu, self._rmin_gpu, self._shape_gpu = [None] * self._n_layers, [None] * self._n_layers, [None] * self._n_layers, [None] * self._n_layers
            self._qt_f_recv, self._scale_recv, self._rmin_recv, self._shape_recv = [None] * self._n_layers, [None] * self._n_layers, [None] * self._n_layers, [None] * self._n_layers
            # 初始化反向传播中的量化和解量化的GPU缓冲区
            self._qt_g_gpu, self._g_scale_gpu, self._g_rmin_gpu, self._g_shape_gpu = [None] * self._n_layers, [None] * self._n_layers, [None] * self._n_layers, [None] * self._n_layers
            self._qt_g_recv, self._g_scale_recv, self._g_rmin_recv, self._g_shape_recv = [None] * self._n_layers, [None] * self._n_layers, [None] * self._n_layers, [None] * self._n_layers


            if STALE_CHECK_2:
                # 初始化ver, 记录发送全精度特征的轮次
                self.ver = [[0 for _ in range(size)] for _ in range(self._n_layers)]
                for i in range(self._n_layers):
                    for j in range(size):
                        if j == rank:
                            self.ver[i][j] = -1
                        else:
                            self.ver[i][j] = 0
                if DEBUG:
                    print(f"ver: {self.ver}")
                self.ver_back = [[0 for _ in range(size)] for _ in range(self._n_layers)]
                for i in range(self._n_layers):
                    for j in range(size):
                        if j == rank:
                            self.ver_back[i][j] = -1
                        else:
                            self.ver_back[i][j] = 0

                            

            # 定义新的tensor，打包了四个需要传输的数据
            self.recv_cpu_tensor = [None] * self._n_layers
            self.send_cpu_tensor = [None] * self._n_layers

            # 反向传播中的打包tensor
            self.g_recv_cpu_tensor = [None] * self._n_layers
            self.g_send_cpu_tensor = [None] * self._n_layers

            for i in range(self._n_layers):
                if i == 0 and use_pp:
                    continue
                tmp1, tmp2, tmp3, tmp4 = [], [], [], []


                # 临时变量用于存储合并的tensor
                tmp_send_cpu_tensor = []
                tmp_recv_cpu_tensor = []

                # 反向传播的临时tensor
                tmp_g_send_cpu_tensor = []
                tmp_g_recv_cpu_tensor = []

                for j in range(size):
                    if j == rank:
                        tmp1.append(None)
                        tmp2.append(None)
                        tmp3.append(None)
                        tmp4.append(None)
                        tmp_send_cpu_tensor.append(None)
                        tmp_recv_cpu_tensor.append(None)
                        tmp_g_send_cpu_tensor.append(None)
                        tmp_g_recv_cpu_tensor.append(None)
                        

                    else:
                        s1 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                        # RECV_SHAPE是一个整数数组
                        s2 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                        tmp1.append(torch.zeros(s1).pin_memory())
                        tmp2.append(torch.zeros(s2).pin_memory())
                        tmp3.append(torch.zeros(s2).pin_memory())
                        tmp4.append(torch.zeros(s1).pin_memory())

                        # 定义尺寸
                        # 应该是这里有问题，没有乘以layersize
                        s_tmp_rmin_cpu = torch.Size([boundary[j].shape[0]])
                        s_tmp_rmin_recv_cpu = torch.Size([f_recv_shape[j]])
                        s_tmp_scale_cpu = torch.Size([boundary[j].shape[0]])
                        s_tmp_scale_recv_cpu = torch.Size([f_recv_shape[j]])
                        s_tmp_shape_cpu = torch.Size([2])
                        s_tmp_shape_recv_cpu = torch.Size([2])

                        qt_send_size = self.calculate_packed_tensor_dimension(boundary[j].shape[0], self._layer_size[i], bits)
                        qt_recv_size = self.calculate_packed_tensor_dimension(f_recv_shape[j], self._layer_size[i], bits)
                        # if DEBUG:
                        #     print(f"send_size in {i} layer {rank} to {j}: {qt_send_size}, recv_size in {i} layer {j} to {rank}: {qt_recv_size}")

                        s_tmp_qt_f_cpu = torch.Size([qt_send_size])
                        s_tmp_qt_f_recv_cpu = torch.Size([qt_recv_size])


                        # 合并tensor
                        # 注意，讲rmin和scale设置成了bfloat16类型
                        tmp_send_cpu_tensor.append(pack_tensors(torch.zeros(s_tmp_rmin_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_scale_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_shape_cpu, dtype=torch.int64, pin_memory=True), torch.zeros(s_tmp_qt_f_cpu, dtype=torch.int8, pin_memory=True)))
                        tmp_recv_cpu_tensor.append(pack_tensors(torch.zeros(s_tmp_rmin_recv_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_scale_recv_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_shape_recv_cpu, dtype=torch.int64, pin_memory=True), torch.zeros(s_tmp_qt_f_recv_cpu, dtype=torch.int8, pin_memory=True)))
                        
                        # 反向传播合并tensor
                        tmp_g_send_cpu_tensor.append(pack_tensors(torch.zeros(s_tmp_rmin_recv_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_scale_recv_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_shape_recv_cpu, dtype=torch.int64, pin_memory=True), torch.zeros(s_tmp_qt_f_recv_cpu, dtype=torch.int8, pin_memory=True)))
                        tmp_g_recv_cpu_tensor.append(pack_tensors(torch.zeros(s_tmp_rmin_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_scale_cpu, dtype=torch.bfloat16, pin_memory=True), torch.zeros(s_tmp_shape_cpu, dtype=torch.int64, pin_memory=True), torch.zeros(s_tmp_qt_f_cpu, dtype=torch.int8, pin_memory=True)))
                        


                        if DEBUG:
                            print(f"send_size in {i} layer {rank} to {j}: {tmp_send_cpu_tensor[j].shape}, recv_size in {i} layer {j} to {rank}: {tmp_recv_cpu_tensor[j].shape}")


                self._feat_cpu[i] = tmp1
                self._f_recv_cpu[i] = tmp3

                if i > 0:
                    self._grad_cpu[i] = tmp2
                    self._b_recv_cpu[i] = tmp4

                # 将合并的tensor赋值给相应的缓冲区
                self.send_cpu_tensor[i] = tmp_send_cpu_tensor
                self.recv_cpu_tensor[i] = tmp_recv_cpu_tensor
                # 反向传播中的打包tensor
                if i > 0:
                    self.g_send_cpu_tensor[i] = tmp_g_send_cpu_tensor
                    self.g_recv_cpu_tensor[i] = tmp_g_recv_cpu_tensor





        self._f_buf = [None] * self._n_layers
        self._f_recv, self._b_recv = [], []
        self._comm_stream, self._corr_stream = torch.cuda.Stream(), torch.cuda.Stream()
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []

        self._backend = backend

        self._f_avg, self._b_avg = [None] * self._n_layers, [None] * self._n_layers
        self._f_recv, self._b_recv = [None] * self._n_layers, [None] * self._n_layers
        self._f_cpu_event, self._b_cpu_event = [None] * self._n_layers, [None] * self._n_layers
        self._f_cuda_event, self._b_cuda_event = [None] * self._n_layers, [None] * self._n_layers


        for i in range(self._n_layers):
            if i == 0 and use_pp:
                continue
            self._f_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda')
            tmp1, tmp2, tmp3, tmp4 = [], [], [], []
            tmp_qt_f_gpu, tmp_scale_gpu, tmp_rmin_gpu, tmp_shape_gpu = [], [], [], []
            tmp_qt_f_recv, tmp_scale_recv, tmp_rmin_recv, tmp_shape_recv = [], [], [], []

            tmp_qt_g_gpu, tmp_g_scale_gpu, tmp_g_rmin_gpu, tmp_g_shape_gpu = [], [], [], []
            tmp_qt_g_recv, tmp_g_scale_recv, tmp_g_rmin_recv, tmp_g_shape_recv = [], [], [], []


            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                    tmp3.append(None)
                    tmp4.append(None)


                    tmp_qt_f_gpu.append(None)
                    tmp_scale_gpu.append(None)
                    tmp_rmin_gpu.append(None)
                    tmp_shape_gpu.append(None)

                    tmp_qt_f_recv.append(None)
                    tmp_scale_recv.append(None)
                    tmp_rmin_recv.append(None)
                    tmp_shape_recv.append(None)

                    tmp_qt_g_gpu.append(None)
                    tmp_g_scale_gpu.append(None)
                    tmp_g_rmin_gpu.append(None)
                    tmp_g_shape_gpu.append(None)

                    tmp_qt_g_recv.append(None)
                    tmp_g_scale_recv.append(None)
                    tmp_g_rmin_recv.append(None)
                    tmp_g_shape_recv.append(None)

                else:
                    s1 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda'))
                    tmp2.append(torch.zeros(s2, device='cuda'))
                    tmp3.append(torch.zeros(s1, device='cuda'))
                    tmp4.append(torch.zeros(s2, device='cuda'))

                    # 定义尺寸
                    s_tmp_rmin_recv = torch.Size([f_recv_shape[j]])
                    s_tmp_scale_recv = torch.Size([f_recv_shape[j]])
                    s_tmp_shape_recv = torch.Size([2])
                    qt_recv_size = self.calculate_packed_tensor_dimension(f_recv_shape[j], self._layer_size[i], bits)

                    s_tmp_rmin_gpu = torch.Size([boundary[j].shape[0]])
                    s_tmp_scale_gpu = torch.Size([boundary[j].shape[0]])
                    s_tmp_shape_gpu = torch.Size([2])
                    qt_send_size = self.calculate_packed_tensor_dimension(boundary[j].shape[0], self._layer_size[i], bits)

                    s_tmp_qt_f_recv = torch.Size([qt_recv_size])


                    s_tmp_qt_f_gpu = torch.Size([qt_send_size])


                    tmp_qt_f_gpu.append(torch.zeros(s_tmp_qt_f_gpu, dtype=torch.int8, device='cuda'))
                    tmp_scale_gpu.append(torch.zeros(s_tmp_scale_gpu, dtype=torch.bfloat16, device='cuda'))
                    tmp_rmin_gpu.append(torch.zeros(s_tmp_rmin_gpu, dtype=torch.bfloat16, device='cuda'))
                    tmp_shape_gpu.append(torch.zeros(s_tmp_shape_gpu, dtype=torch.int64, device='cuda'))

                    # 反向传播的缓冲区刚好跟前向传播相反
                    tmp_qt_g_gpu.append(torch.zeros(s_tmp_qt_f_recv, dtype=torch.int8, device='cuda'))
                    tmp_g_scale_gpu.append(torch.zeros(s_tmp_scale_recv, dtype=torch.bfloat16, device='cuda'))
                    tmp_g_rmin_gpu.append(torch.zeros(s_tmp_rmin_recv, dtype=torch.bfloat16, device='cuda'))
                    tmp_g_shape_gpu.append(torch.zeros(s_tmp_shape_recv, dtype=torch.int64, device='cuda'))

                    tmp_qt_f_recv.append(torch.zeros(s_tmp_qt_f_recv, dtype=torch.int8, device='cuda'))
                    tmp_scale_recv.append(torch.zeros(s_tmp_scale_recv, dtype=torch.bfloat16, device='cuda'))
                    tmp_rmin_recv.append(torch.zeros(s_tmp_rmin_recv, dtype=torch.bfloat16, device='cuda'))
                    tmp_shape_recv.append(torch.zeros(s_tmp_shape_recv, dtype=torch.int64, device='cuda'))

                    # 反向传播
                    tmp_qt_g_recv.append(torch.zeros(s_tmp_qt_f_gpu, dtype=torch.int8, device='cuda'))
                    tmp_g_scale_recv.append(torch.zeros(s_tmp_scale_gpu, dtype=torch.bfloat16, device='cuda'))
                    tmp_g_rmin_recv.append(torch.zeros(s_tmp_rmin_gpu, dtype=torch.bfloat16, device='cuda'))
                    tmp_g_shape_recv.append(torch.zeros(s_tmp_shape_gpu, dtype=torch.int64, device='cuda'))

            self._f_recv[i] = tmp1
            if i > 0:
                self._b_recv[i] = tmp2

            # 暂时不用管校正流
            if corr_feat:
                self._f_avg[i] = tmp3
            if corr_grad and i > 0:
                self._b_avg[i] = tmp4
            
            # 发送缓冲区初始化
            self._qt_f_gpu[i] = tmp_qt_f_gpu
            self._scale_gpu[i] = tmp_scale_gpu
            self._rmin_gpu[i] = tmp_rmin_gpu
            self._shape_gpu[i] = tmp_shape_gpu

            # 反向传播中的发送缓冲区初始化
            if i > 0:
                self._qt_g_gpu[i] = tmp_qt_g_gpu
                self._g_scale_gpu[i] = tmp_g_scale_gpu
                self._g_rmin_gpu[i] = tmp_g_rmin_gpu
                self._g_shape_gpu[i] = tmp_g_shape_gpu


            # 接收缓冲区初始化
            self._qt_f_recv[i] = tmp_qt_f_recv
            self._scale_recv[i] = tmp_scale_recv
            self._rmin_recv[i] = tmp_rmin_recv
            self._shape_recv[i] = tmp_shape_recv

            # 反向传播中的接收缓冲区初始化
            if i > 0:
                self._qt_g_recv[i] = tmp_qt_g_recv
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
                time_durations = {}

                if Count_Forward_Time and rank == 0:
                    overall_start = time.time()

                if Count_Time and rank == 0:
                    start_time = time.time()
                    print(f"start transfer*** epoch: {epoch} layer: {layer} start time: {start_time}")

                if DEBUG:
                    print("gloo start")
                
                req_send, req_quantization = [], queue.Queue()
                self._comm_stream.wait_stream(torch.cuda.current_stream()) 
                if DEBUG:
                    print("prepare send")
                with torch.cuda.stream(self._comm_stream):
                    if DEBUG:
                        print("prepare quantization")
                    with qt_timer.timer(f"forward_qt_{epoch}_{layer}"):
                        for i in range(1, size):
                            right = (rank + i) % size
                            try:
                                if epoch >= 0:
                                    shape = message_quantization(feat[self._boundary[right]], bits, stochastic=True, 
                                                                q_input_holder=self._qt_f_gpu[layer][right], 
                                                                q_scale_holder=self._scale_gpu[layer][right], 
                                                                rmin_holder=self._rmin_gpu[layer][right])
                                    if DEBUG:
                                        print(f"shape:{shape}")
                                    self._shape_gpu[layer][right] = torch.tensor(shape)
                                    del shape
                                if DEBUG:
                                    print(f"shape after tensor:{self._shape_gpu[layer][right]}")
                            except Exception as e:
                                if DEBUG:
                                    print(f"Quantization error: {e}")
                                raise

                    quantization_end = time.time()
                    if Count_Forward_Time and rank == 0:
                        time_durations['Quantization'] = quantization_end - overall_start

                    if DEBUG:
                        print("finish quantization")
                    if DEBUG:
                        print("prepare for")

                    pack_time_total = 0
                    unpack_time_total = 0
                    dequantization_time_total = 0

                    for i in range(1, size):
                        left = (rank - i + size) % size
                        right = (rank + i) % size

                        pack_start_time = time.time()
                        with pack_timer.timer(f"forward_pack_{epoch}_{layer}_{i}"):
                            try:
                                # 总共5.5s左右
                                # 这里1.4s
                                if epoch > -1:
                                    tmp_send_gpu_tensor = pack_tensors(self._rmin_gpu[layer][right], self._scale_gpu[layer][right], self._shape_gpu[layer][right], self._qt_f_gpu[layer][right])
                                # self.send_cpu_tensor[layer][right].copy_(tmp_send_gpu_tensor, non_blocking=True)
                                # 这里0.5s
                                    self.send_cpu_tensor[layer][right].copy_(tmp_send_gpu_tensor)
                                    del tmp_send_gpu_tensor
                                # torch.cuda.empty_cache()

                            except Exception as e:
                                if DEBUG:
                                    print(f"Combine error: {e}")
                                raise

                        pack_end_time = time.time()
                        pack_time_total += pack_end_time - pack_start_time

                        try:
                            r1 = dist.isend(self.send_cpu_tensor[layer][right], tag=tag, dst=right)
                            req_send.append(r1)
                            r2 = dist.irecv(self.recv_cpu_tensor[layer][left], tag=tag, src=left)
                            req_quantization.put((r2, left))  

                        except Exception as e:
                            if DEBUG:
                                print(f"Send/Receive error: {e}")

                    send_receive_end = time.time()
                    if Count_Forward_Time and rank == 0:
                        time_durations['Send/Receive'] = send_receive_end - quantization_end

                    while not req_quantization.empty():
                        r, idx = req_quantization.get()
                        r.wait()
                        unpack_start_time = time.time()
                        try:
                            with unpack_timer.timer(f"forward_unpack_{epoch}_{layer}_{idx}"):
                                if epoch > -1:
                                    recv_rmin, recv_scale, recv_shape, recv_q_feat = unpack_tensors(self.recv_cpu_tensor[layer][idx])
                            unpack_end_time = time.time()
                            
                            unpack_time_total += unpack_end_time - unpack_start_time
                            # 这里可以加一段时间统计
                            if epoch > -1:
                                self._qt_f_recv[layer][idx] = recv_q_feat.to('cuda')
                                self._scale_recv[layer][idx] = recv_scale.to('cuda')
                                self._rmin_recv[layer][idx] = recv_rmin.to('cuda')
                                self._shape_recv[layer][idx] = recv_shape.to('cuda')
                            # self._qt_f_recv[layer][idx].copy_(recv_q_feat)
                            # self._scale_recv[layer][idx].copy_(recv_scale)
                            # self._rmin_recv[layer][idx].copy_(recv_rmin)
                            # self._shape_recv[layer][idx].copy_(recv_shape)
                            shape_tensor = self._shape_recv[layer][idx]
                            if shape_tensor[1].item() == 256:
                                shape = (shape_tensor[0].item(), shape_tensor[1].item())
                            else:
                                shape = (shape_tensor[1].item(), 256)
                            
                            dequantization_start_time = time.time()
                            with dq_timer.timer(f"forward_dq_{epoch}_{layer}_{idx}"):
                                if epoch > -1:
                                    self._f_recv[layer][idx] = message_dequantization(self._qt_f_recv[layer][idx], self._scale_recv[layer][idx], self._rmin_recv[layer][idx], shape, bits)
                            dequantization_end_time = time.time()
                            dequantization_time_total += dequantization_end_time - dequantization_start_time
                            if epoch > -1:
                                del recv_rmin, recv_scale, recv_shape, recv_q_feat, shape_tensor
                            torch.cuda.empty_cache()

                        except Exception as e:
                            if DEBUG:
                                print(f"Dequantization error: {e}")
                            raise

                    for r in req_send:
                        r.wait()

                    dequantization_end = time.time()
                    if Count_Forward_Time and rank == 0:
                        time_durations['Pack Total'] = pack_time_total
                        time_durations['Unpack Total'] = unpack_time_total
                        time_durations['Dequantization Total'] = dequantization_time_total
                        time_durations['Dequantization'] = dequantization_end - send_receive_end

                if Count_Time and rank == 0:
                    end_time = time.time()
                    print(f"end transfer*** epoch: {epoch} layer: {layer} end time: {end_time} cousume time: {end_time - start_time}")
                
                if Count_Forward_Time and rank == 0:
                    total_duration = time.time() - overall_start
                    print(f"Total operation time: {total_duration} seconds")
                    for key, duration in time_durations.items():
                        print(f"{key} duration: {duration:.4f} seconds")
                    print("***")


# 反向传播量化
    def __qt_backward_gloo_all_to_all(self, grad, layer, tag, bits, epoch, forward=False):
        with true_comm_timer.timer(f"backward_{epoch}__{layer}"):
            with backward_comm_timer.timer(f"_{epoch}_{layer}"):
                if DEBUG:
                    print("gloo backward start")
                rank, size = dist.get_rank(), dist.get_world_size()
                req_send, req_quantization = [], queue.Queue()
                if Count_Time and rank == 0:
                    start_time = time.time()
                    print(f"start backward transfer*** epoch: {epoch} layer: {layer} start time: {start_time}")

                # 在启动通信流 _comm_stream 之前，确保当前默认流中的所有操作已经完成。
                self._comm_stream.wait_stream(torch.cuda.current_stream()) 
                if DEBUG:
                    print("backward prepare send")
                with torch.cuda.stream(self._comm_stream):
                    if DEBUG:
                        print("backward prepare quantization")
                    with qt_timer.timer(f"backward_qt_{epoch}_{layer}"):
                        for i in range(1, size):
                            right = (rank + i) % size
                            try:
                                
                                if DEBUG:
                                    # 这一步出问题，grad[self._pl[right]:self._pr[right]]打印不出来
                                    print(f"GRAD:{grad[self._pl[right]:self._pr[right]]}")
                                    print(f"QT_G_GPU:{self._qt_g_gpu[layer][right]}")
                                    print(f"backward layer{layer}-dst{right}: grad shape->{grad[self._pl[right]:self._pr[right]].shape}")
                                if epoch >= 0:
                                    shape = message_quantization(grad[self._pl[right]:self._pr[right]], bits, stochastic=True, 
                                                                q_input_holder=self._qt_g_gpu[layer][right], 
                                                                q_scale_holder=self._g_scale_gpu[layer][right], 
                                                                rmin_holder=self._g_rmin_gpu[layer][right])                                    
                                    self._g_shape_gpu[layer][right] = torch.tensor(shape)
                                    del shape
                            except Exception as e:
                                if DEBUG:
                                    print(f"backward Quantization error: {e}")
                                raise
                    if DEBUG:
                        print("backward finish quantization")
                    if DEBUG:
                        print("backward prepare for")
                    for i in range(1, size):
                        left = (rank - i + size) % size
                        right = (rank + i) % size

                        try:
                            # if DEBUG:
                            #     print(f"{feat[self._boundary[right]]}")
                            # 确认索引正确
                            if right > len(self._qt_g_gpu[layer]) or right > len(self._boundary):
                                raise IndexError(f"backward Index out of range: right={right}, len(self._qt_g_gpu)={len(self._qt_g_gpu)}, len(self._boundary)={len(self._boundary)}")            


                            if DEBUG:
                                print(f"backward layer{layer}, src{rank}, dst{right} before pack buffer: rmin:{self._g_rmin_gpu[layer][right]}, scale:{self._g_scale_gpu[layer][right]}, shape:{self._g_shape_gpu[layer][right]}, qt:{self._qt_g_gpu[layer][right]}")
                                print(f"backward layer{layer}, src{rank}, dst{right} before pack_tensors: rmin_shape:{self._g_rmin_gpu[layer][right].shape}, scale_shape:{self._g_scale_gpu[layer][right].shape}, shape_shape:{self._g_shape_gpu[layer][right].shape}, qt_shape:{self._qt_g_gpu[layer][right].shape}")
                            with pack_timer.timer(f"backward_pack_{epoch}_{layer}_{i}"):
                                if epoch > -1:
                                    tmp_send_gpu_tensor = pack_tensors(self._g_rmin_gpu[layer][right], self._g_scale_gpu[layer][right], self._g_shape_gpu[layer][right], self._qt_g_gpu[layer][right])
                                    if DEBUG:
                                        print(f"g_send_cpu_tensor size: {self.g_send_cpu_tensor[layer][right].shape}, tmp size: {tmp_send_gpu_tensor.shape}")
                                    # self.g_send_cpu_tensor[layer][right].copy_(tmp_send_gpu_tensor, non_blocking=True)
                                    self.g_send_cpu_tensor[layer][right].copy_(tmp_send_gpu_tensor)
                            # 释放显存
                                    del tmp_send_gpu_tensor
                            # torch.cuda.empty_cache()

                            # self._grad_cpu[layer][right].copy_(feat[self._boundary[right]])


                        except Exception as e:
                            if DEBUG:
                                print(f"backward Combine error: {e}")
                            raise

                        try:
                            if DEBUG:
                                print("backward begin try")

                            r1 = dist.isend(self.g_send_cpu_tensor[layer][right], tag=tag, dst=right)
                            

                            req_send.append(r1)
                            
                            if DEBUG:
                                print("backward send ok!")
                            
                            # 打印数据长度，与缓冲区大小进行对比
                            if DEBUG:

                                print(f"backward {rank} send to {right}: {self.g_send_cpu_tensor[layer][right].shape}, {left} send to {rank}: {self.g_recv_cpu_tensor[layer][left].shape}")                     
                            
                            if DEBUG:
                                print(f"backward {rank} recv {left} shape: {self.g_recv_cpu_tensor[layer][left].shape}")

                            # recv ok的数量 + Send/Receive error的数量 = begin try的数量
                            r2 = dist.irecv(self.g_recv_cpu_tensor[layer][left], tag=tag, src=left)
                            if DEBUG:
                                print("backward recv ok!!!")
                            req_quantization.put((r2, left))  
                            if DEBUG:
                                print("backward put r2 ok")
                        # 这步有报错，可能是读/写冲突导致的
                        except Exception as e:
                            if DEBUG:
                                print(f"backward Send/Receive error: {e}, at {layer} layer, {rank} send  to {right}: {self.g_send_cpu_tensor[layer][right].shape}, {rank} recv {left} shape: {self.g_recv_cpu_tensor[layer][left].shape}")
                            # raise

                    while not req_quantization.empty():
                        if DEBUG:
                            print("backward prepare address recv!")
                        r, idx = req_quantization.get()
                        if DEBUG:
                            print("backward get req_quantization success")
                            print(r)
                            # print(self.g_recv_cpu_tensor[layer][idx])
                        
                        r.wait()
                        try:
                            if DEBUG:
                                print("backward address begin!")

                            # 解包打包的张量
                            with unpack_timer.timer(f"backward_unpack_{epoch}_{layer}_{idx}"):
                                if epoch > -1:
                                    recv_rmin, recv_scale, recv_shape, recv_q_feat = unpack_tensors(self.g_recv_cpu_tensor[layer][idx])
                            # 解量化接收到的数据
                            if epoch > -1:
                                self._qt_g_recv[layer][idx] = recv_q_feat.to('cuda')
                                self._g_scale_recv[layer][idx] = recv_scale.to('cuda')
                                self._g_rmin_recv[layer][idx] = recv_rmin.to('cuda')
                                self._g_shape_recv[layer][idx] = recv_shape.to('cuda')
                            # self._qt_g_recv[layer][idx].copy_(recv_q_feat)
                            # self._g_scale_recv[layer][idx].copy_(recv_scale)
                            # self._g_rmin_recv[layer][idx].copy_(recv_rmin)
                            # self._g_shape_recv[layer][idx].copy_(recv_shape)


                            shape_tensor = self._g_shape_recv[layer][idx]
                            shape = (shape_tensor[0].item(), shape_tensor[1].item())
                            with dq_timer.timer(f"backward_dq_{epoch}_{layer}_{idx}"):
                                if epoch > -1:
                                    self._b_recv[layer][idx] = message_dequantization(self._qt_g_recv[layer][idx], self._g_scale_recv[layer][idx], self._g_rmin_recv[layer][idx], shape, bits)
                            # 释放显存
                            if epoch > -1:
                                del recv_rmin, recv_scale, recv_shape, recv_q_feat, shape_tensor
                            torch.cuda.empty_cache()
                            if DEBUG:
                                print("backward address ok!")
                        except Exception as e:
                            if DEBUG:
                                print(f"backward Dequantization error: {e}")
                            raise
                    
                    # TODO: remove this 'wait'
                    for r in req_send:
                        r.wait()     
                if Count_Time and rank == 0:
                    end_time = time.time()
                    print(f"end backward transfer*** epoch: {epoch} layer: {layer} end time: {end_time} cousume time: {end_time - start_time}")       

# 前向传播，不使用量化
    def __forward_gloo_all_to_all(self, feat, layer, tag, epoch, forward=True):
        with true_comm_timer.timer(f'forward_{epoch}_{layer}'):
            with forward_comm_timer.timer(f'_{epoch}_{layer}'):
                rank, size = dist.get_rank(), dist.get_world_size()

                if Count_Forward_Time and rank == 0:
                    overall_start = time.time()

                if Count_Time and rank == 0:
                    start_time = time.time()
                    print(f"start transfer*** epoch: {epoch} layer: {layer} start time: {start_time}")

                
                req_send, req_quantization = [], queue.Queue()
                self._comm_stream.wait_stream(torch.cuda.current_stream()) 
                with torch.cuda.stream(self._comm_stream):
                    for i in range(1, size):
                        left = (rank - i + size) % size
                        right = (rank + i) % size
                        self._feat_cpu[layer][right].copy_(feat[self._boundary[right]])

                        try:
                            r1 = dist.isend(self._feat_cpu[layer][right], tag=tag, dst=right)
                            req_send.append(r1)
                            r2 = dist.irecv(self._f_recv_cpu[layer][left], tag=tag, src=left)
                            req_quantization.put((r2, left))  

                        except Exception as e:
                            if DEBUG:
                                print(f"Send/Receive error: {e}")

                    while not req_quantization.empty():
                        r, idx = req_quantization.get()
                        r.wait()
                        self._f_recv[layer][idx].copy_(self._f_recv_cpu[layer][idx], non_blocking=True)

                    for r in req_send:
                        r.wait()


                if Count_Time and rank == 0:
                    end_time = time.time()
                    print(f"end transfer*** epoch: {epoch} layer: {layer} end time: {end_time} cousume time: {end_time - start_time}")
                
                if Count_Forward_Time and rank == 0:
                    total_duration = time.time() - overall_start
                    print(f"Total operation time: {total_duration} seconds")


# 反向传播, 不使用量化
    def __backward_gloo_all_to_all(self, grad, layer, tag, epoch, forward=False):
        with true_comm_timer.timer(f"backward_{epoch}__{layer}"):
            with backward_comm_timer.timer(f"_{epoch}_{layer}"):
                rank, size = dist.get_rank(), dist.get_world_size()
                req_send, req_quantization = [], queue.Queue()
                if Count_Time and rank == 0:
                    start_time = time.time()
                    print(f"start backward transfer*** epoch: {epoch} layer: {layer} start time: {start_time}")

                # 在启动通信流 _comm_stream 之前，确保当前默认流中的所有操作已经完成。
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
                        req_quantization.put((r2, left))  


                    while not req_quantization.empty():
                        
                        r, idx = req_quantization.get()
                        r.wait()
                        self._b_recv[layer][idx].copy_(self._b_recv_cpu[layer][idx], non_blocking = True)

                    # TODO: remove this 'wait'
                    for r in req_send:
                        r.wait()     
                if Count_Time and rank == 0:
                    end_time = time.time()
                    print(f"end backward transfer*** epoch: {epoch} layer: {layer} end time: {end_time} cousume time: {end_time - start_time}")       



    def next_epoch(self):
        self._epoch += 1

    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                if self._corr_feat:
                    tmp.append(self._f_avg[layer][i])
                else:
                    # 这里维度有问题，应该是量化时的问题
                    tmp.append(self._f_recv[layer][i])
                    
                    # print(f"self._f_recv[{layer}][{i}]: {self._f_recv[layer][i].shape}")
        
        # print(f"feat: {feat.shape}")
        return torch.cat(tmp)
    
    def get_group(self, epoch, layer, _type):
        key = (epoch % GROUP_NUM, layer, _type)
        if key not in self._groups:
            # dist.barrier()
            if TRANFER_DEBUG:
                print(f"Creating new group for epoch: {epoch}, layer: {layer}, type: {_type}")
            ranks = list(range(dist.get_world_size()))  # 假设所有进程参与通信
            self._groups[key] = dist.new_group(ranks=ranks)
        else:
            if TRANFER_DEBUG:
                print(f"Using existing group for epoch: {epoch}, layer: {layer}")
        return self._groups[key]
    # def get_group(self, epoch, layer):

    #     ranks = list(range(dist.get_world_size()))  # 假设所有进程参与通信

    #     return dist.new_group(ranks=ranks)
    
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
            with comm_timer.timer(f'forward_{layer}'):
                self.__feat_transfer(epoch, layer, feat, bits, f_transfer_group)
                # 位置：在依赖通信流操作的后续计算操作之前。
                # 作用：让当前默认流等待事件 self._f_cuda_event[layer] 的完成。
                # 目的：确保当前默认流中的操作在 self._f_cuda_event[layer] 事件触发后才开始执行，即等待通信流中的特定操作完成。
                torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])

            self._f_buf[layer] = self.__feat_concat(layer, feat)
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(epoch, layer, bits, b_transfer_group, b_update_group, rank))
            # return True, self._f_buf[layer]
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
                        
                        k = (epoch * 3 + layer - 1) / self.f_cnt
                        if Count_Stale:
                            print(f"***k:{k} epoch:{epoch} layer: {layer}")
                        # 考虑记入true comm time
                        signals = self.forward_signals
                        # Rank 0 进程进行判断
                        if rank == 0:
                            condition_met = self._f_cpu_event[layer].is_set() or k >= STALE_THRESHOLD
                            signals[layer] = torch.tensor(int(condition_met), dtype=torch.int)
                        # print(f"signal: {signals[layer]} epoch: {epoch} layer: {layer}")
                        # 广播信号给所有进程
                        dist.broadcast(signals[layer], src=0, group=f_update_group)                            

                        # dist.barrier(f_update_group)
                        if signals[layer].item() == 1:
                            if Count_Stale:
                                print(f"forward transfer {epoch} {layer}")
                            self.f_cnt += 1
                            # self._f_cpu_event[layer].wait()    
                            # torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
                            self._f_cpu_event[layer].clear()
                        else:
                            self._f_buf[layer] = self.__feat_concat(layer, feat)
                            return self._f_buf[layer]
                    # if STALE:
                    #     # 计算陈旧阈值
                    #     k = (epoch * 3 + layer - 1) / self.f_cnt
                    #     if Count_Stale:
                    #         print(f"***k:{k} epoch:{epoch} layer: {layer}")
                    #     if k >= STALE_THRESHOLD:
                    #         self.f_cnt += 1
                    #         if Count_Stale:
                    #             print(f"forward transfer {epoch} {layer}")
                    #         # dist.barrier(f_group)
                    #         self._f_cpu_event[layer].wait()
                    #         torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
                    #         self._f_cpu_event[layer].clear()
                    #     else:
                    #         dist.barrier(f_group)
                    #         if Count_Stale:
                    #             print(f"forward no transfer {epoch} {layer}")
                    #         if self._f_cpu_event[layer].is_set() == False:
                    #             # return  False, []
                    #             self._f_buf[layer] = self.__feat_concat(layer, feat)
                    #             return self._f_buf[layer]
                    #         else:
                    #             self.f_cnt += 1
                    #             self._f_cpu_event[layer].clear()
                    else:
                        self._f_cpu_event[layer].wait()
                        torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
                        self._f_cpu_event[layer].clear()
            # BNS-GCN的这一步在传输后面，可见是先计算再传播
            # 第二次到这的时候卡了，应该是
            self._f_buf[layer] = self.__feat_concat(layer, feat)
            # 使用异步线程池 _pool 进行异步特征传输，以提高效率
            self._pool.apply_async(self.__feat_transfer, args=(epoch, layer, feat, bits, f_transfer_group))
            # 通过注册梯度钩子 __grad_hook，确保梯度在反向传播过程中正确传递和更新
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(epoch, layer, bits, b_transfer_group, b_update_group, rank))
            # return True, self._f_buf[layer]
            return self._f_buf[layer]





    def __gloo_all_to_all(self, send_gpu, send_cpu, recv_cpu, recv_gpu, tag, corr, epoch, layer, avg=None, forward=True):
        with true_comm_timer.timer(f"forward: {forward}_{epoch}__{layer}"):
            if forward:
                with forward_comm_timer.timer(f"_{epoch}_{layer}"):
                    if DEBUG:
                        print("Start grad transfer")

                    rank, size = dist.get_rank(), dist.get_world_size()
                    if Count_Time and rank == 0:
                        start_time = time.time()
                        print(f"start transfer*** forward: {forward}, epoch: {epoch}, layer: {layer}, start time: {start_time}")

                    req_send, req_quantization = [], queue.Queue()
                    
                    # 同步默认流，确保前面的操作都已完成
                    if DEBUG:
                        print("Synchronizing default stream before starting communication stream")
                    self._comm_stream.wait_stream(torch.cuda.current_stream())

                    # 在通信流上下文中执行后续操作
                    with torch.cuda.stream(self._comm_stream):
                        for i in range(1, size):
                            left = (rank - i + size) % size
                            right = (rank + i) % size
                            if DEBUG:
                                print(f"Starting irecv from rank {left} with tag {tag}")
                            r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
                            req_quantization.put((r2, left))
                            if forward:
                                if DEBUG:
                                    print(f"Copying data for forward transfer to send_cpu[{right}] from send_gpu")
                                send_cpu[right].copy_(send_gpu[self._boundary[right]])
                            else:
                                if DEBUG:
                                    print(f"Copying data for backward transfer to send_cpu[{right}] from send_gpu")
                                    print(f"{send_gpu[self._pl[right]:self._pr[right]]}")
                                send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]])
                            if DEBUG:
                                print(f"Starting isend to rank {right} with tag {tag}")
                            r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
                            req_send.append(r1)

                        while not req_quantization.empty():
                            r, idx = req_quantization.get()
                            if DEBUG:
                                print(f"Waiting for irecv from rank {idx}")
                            r.wait()
                            if DEBUG:
                                print(f"Copying received data to recv_gpu[{idx}]")
                            recv_gpu[idx].copy_(recv_cpu[idx], non_blocking=True)
                            if corr:
                                if DEBUG:
                                    print("Starting correction operation in correction stream")
                                with torch.cuda.stream(self._corr_stream):
                                    if DEBUG:
                                        print("Synchronizing correction stream with communication stream")
                                    self._corr_stream.wait_stream(self._comm_stream)
                                    t = avg[idx]
                                    t *= self._corr_momentum
                                    t += (1 - self._corr_momentum) * recv_gpu[idx]

                        if DEBUG:
                            print("Waiting for all isend requests to complete")
                        for r in req_send:
                            r.wait()

                    if DEBUG:
                        print("End grad transfer")

                    if Count_Time and rank == 0:
                        end_time = time.time()
                        print(f"end transfer*** forward{forward}, epoch: {epoch}, layer: {layer}, end time: {end_time} cousume time: {end_time - start_time}")
            else:
                with backward_comm_timer.timer(f"_{epoch}_{layer}"):
                    if DEBUG:
                        print("Start grad transfer")

                    rank, size = dist.get_rank(), dist.get_world_size()
                    if Count_Time and rank == 0:
                        start_time = time.time()
                        print(f"start transfer*** forward: {forward}, epoch: {epoch}, layer: {layer}, start time: {start_time}")

                    req_send, req_quantization = [], queue.Queue()
                    
                    # 同步默认流，确保前面的操作都已完成
                    if DEBUG:
                        print("Synchronizing default stream before starting communication stream")
                    self._comm_stream.wait_stream(torch.cuda.current_stream())

                    # 在通信流上下文中执行后续操作
                    with torch.cuda.stream(self._comm_stream):
                        for i in range(1, size):
                            left = (rank - i + size) % size
                            right = (rank + i) % size
                            if DEBUG:
                                print(f"Starting irecv from rank {left} with tag {tag}")
                            r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
                            req_quantization.put((r2, left))
                            if forward:
                                if DEBUG:
                                    print(f"Copying data for forward transfer to send_cpu[{right}] from send_gpu")
                                send_cpu[right].copy_(send_gpu[self._boundary[right]])
                            else:
                                if DEBUG:
                                    print(f"Copying data for backward transfer to send_cpu[{right}] from send_gpu")
                                    print(f"{send_gpu[self._pl[right]:self._pr[right]]}")
                                send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]])
                            if DEBUG:
                                print(f"Starting isend to rank {right} with tag {tag}")
                            r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
                            req_send.append(r1)

                        while not req_quantization.empty():
                            r, idx = req_quantization.get()
                            if DEBUG:
                                print(f"Waiting for irecv from rank {idx}")
                            r.wait()
                            if DEBUG:
                                print(f"Copying received data to recv_gpu[{idx}]")
                            recv_gpu[idx].copy_(recv_cpu[idx], non_blocking=True)
                            if corr:
                                if DEBUG:
                                    print("Starting correction operation in correction stream")
                                with torch.cuda.stream(self._corr_stream):
                                    if DEBUG:
                                        print("Synchronizing correction stream with communication stream")
                                    self._corr_stream.wait_stream(self._comm_stream)
                                    t = avg[idx]
                                    t *= self._corr_momentum
                                    t += (1 - self._corr_momentum) * recv_gpu[idx]

                        if DEBUG:
                            print("Waiting for all isend requests to complete")
                        for r in req_send:
                            r.wait()

                    if DEBUG:
                        print("End grad transfer")

                    if Count_Time and rank == 0:
                        end_time = time.time()
                        print(f"end transfer*** forward{forward}, epoch: {epoch}, layer: {layer}, end time: {end_time} cousume time: {end_time - start_time}")
        
                


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
            if FORWARD_QUANTIZATION and not STALE_CHECK_2:
                if TRANFER_DEBUG:
                    print(f"Forward quantization without stale check for layer: {layer}")
                self.__qt_forward_gloo_all_to_all(feat, layer, tag, bits, epoch)
            elif FORWARD_QUANTIZATION and STALE_CHECK_2:
                if TRANFER_DEBUG:
                    print(f"Forward quantization with stale check for layer: {layer}")
                self.__stale2_forward_gloo_all_to_all(feat, layer, tag, bits, epoch)
            else:
                if TRANFER_DEBUG:
                    print(f"Standard all-to-all communication for layer: {layer}")
                self.__forward_gloo_all_to_all(feat, layer, tag, epoch)
                # self.__gloo_all_to_all(feat, self._feat_cpu[layer], self._f_recv_cpu[layer], self._f_recv[layer],
                #                     tag, self._corr_feat, epoch, layer, self._f_avg[layer], forward=True)
            
            self._f_cuda_event[layer].record(self._comm_stream)
            if self._corr_feat:
                self._f_cuda_event[layer].record(self._corr_stream)
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
                # 好像这里卡了？
                dist.barrier(group)


        self._f_cpu_event[layer].set()
        if TRANFER_DEBUG:
            print(f"CPU event set for layer: {layer}")

    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i == rank:
                continue
            else:
                if self._corr_grad:
                    grad[self._boundary[i]] += self._b_avg[layer][i]
                else:
                    grad[self._boundary[i]] += self._b_recv[layer][i]

    # 类比Update，在update中被注册，在Loss.backward中被调用
    # 应当不需要进行修改，只需要修改grad_transfer即可
    def __grad_hook(self, epoch, layer, bits, transfer_group, update_group, rank):
        def fn(grad):
            if WAIT_REDUCE:
                torch.cuda.current_stream().synchronize()

            if self._pipeline is False:
                with comm_timer.timer(f'backward_{layer}'):
                    self.__grad_transfer(epoch, layer, grad, bits, transfer_group)
                    torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                self.__update_grad(layer, grad)
                return grad
            else:
                if self._epoch > 0:
                    with comm_timer.timer(f'backward_{layer}'):
                        if STALE:
                            # dist.barrier(group)
                            k = (epoch * 3 + layer - 1) / self.b_cnt
                            signals = self.backward_signals
                            if rank == 0:
                                condition_met = self._b_cpu_event[layer].is_set() or k >= STALE_THRESHOLD
                                signals[layer] = torch.tensor(int(condition_met), dtype=torch.int)
                            dist.broadcast(signals[layer], src=0, group=update_group)



                            if signals[layer].item() == 0:
                                self.__update_grad(layer, grad)
                                return grad
                            else:
                                if Count_Stale:
                                    print(f"forward transfer {epoch} {layer}")
                                self.b_cnt += 1
                                # self._f_cpu_event[layer].wait()    
                                # torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
                                self._b_cpu_event[layer].clear()
                            # if k >= STALE_THRESHOLD:
                            #     self.b_cnt += 1
                            #     if Count_Stale:
                            #             print(f"backward transfer{epoch} {layer}")
                            #     self._b_cpu_event[layer].wait()
                            #     # torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                            #     self._b_cpu_event[layer].clear()
                            # else:
                            #     if self._b_cpu_event[layer].is_set() == False:
                                    
                            #         # dist.barrier(group)
                            #         if Count_Stale:
                            #             print(f"backward no transfer{epoch} {layer}")
                            #         self.__update_grad(layer, grad)
                            #         return grad
                            #     else:
                            #         self.b_cnt += 1
                            #         self._b_cpu_event[layer].clear()
                        else:
                            
                            self._b_cpu_event[layer].wait()
                            torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                            self._b_cpu_event[layer].clear()

                self.__update_grad(layer, grad)
                self._pool.apply_async(self.__grad_transfer, args=(epoch, layer, grad, bits, transfer_group))
                return grad
        return fn

    # 类比feat_transfer
    def __grad_transfer(self, epoch, layer, grad, bits, group):
        tag = epoch * 4 * self._n_layers + layer + self._n_layers * 2

        if self._backend == 'gloo':
            if BACKWARD_QUANTIZATION and not STALE_CHECK_2:
                self.__qt_backward_gloo_all_to_all(grad, layer, tag, bits, epoch)
            elif BACKWARD_QUANTIZATION and STALE_CHECK_2:
                self.__stale2_backward_gloo_all_to_all(grad, layer, tag, bits, epoch)
            else:
                self.__backward_gloo_all_to_all(grad, layer, tag, epoch)
                # self.__gloo_all_to_all(grad, self._grad_cpu[layer], self._b_recv_cpu[layer], self._b_recv[layer],
                #                    tag, self._corr_grad, epoch, layer, self._b_avg[layer], forward=False)
            self._b_cuda_event[layer].record(self._comm_stream)
            if self._corr_grad:
                self._b_cuda_event[layer].record(self._corr_stream)
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

