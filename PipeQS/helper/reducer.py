import torch
import torch.distributed as dist
from multiprocessing.pool import ThreadPool
from config import *
import time

class Reducer(object):

    def __init__(self):
        super(Reducer, self).__init__()
        self._data_cpu = {}
        self._pool = None
        self._handles = []
        self._stream = None

    def init(self, model):
        cnt = 0
        for i, (name, param) in enumerate(model.named_parameters()):
            cnt += 1
            self._data_cpu[name] = (torch.zeros_like(param.data, pin_memory=True, device='cpu'), dist.new_group())
        self._pool = ThreadPool(processes=cnt)
        self._stream = torch.cuda.Stream()

    def reduce(self, rank, param, name, data, n_train):
        def create_stream():        
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                data.div_(n_train)
                data_cpu, group = self._data_cpu[name]
                data_cpu.copy_(data)  # 明确使用非阻塞拷贝
                dist.all_reduce(data_cpu, op=dist.ReduceOp.SUM, group=group)
                param.grad.copy_(data_cpu, non_blocking=True) 

        self._handles.append(self._pool.apply_async(create_stream))

    def synchronize(self, epoch, rank):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
        torch.cuda.current_stream().wait_stream(self._stream)
