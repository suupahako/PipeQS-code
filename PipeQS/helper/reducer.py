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
            if Count_Time and rank == 0:
                reduce_start_time = time.time()
                print(f"reduce create time*** {reduce_start_time}")            
            self._stream.wait_stream(torch.cuda.current_stream())

            if Count_Time and rank == 0:
                reduce_waitok_time = time.time()
                print(f"reduce wait ok*** {reduce_waitok_time}")               
            with torch.cuda.stream(self._stream):
                data.div_(n_train)
                data_cpu, group = self._data_cpu[name]
                data_cpu.copy_(data)  # 明确使用非阻塞拷贝
                dist.all_reduce(data_cpu, op=dist.ReduceOp.SUM, group=group)
                param.grad.copy_(data_cpu, non_blocking=True)
            if Count_Time and rank == 0:
                reduce_finish_time = time.time()
                print(f"reduce finish time*** {reduce_finish_time}")   

        self._handles.append(self._pool.apply_async(create_stream))
    # ！
    # reduce时间统计有问题，计算了等待compute的时间
    def synchronize(self, epoch, rank):
        # 只保证了create_stream在cpu上完成，不保证其中异步发送的也完成了
        if Count_Time and rank == 0:
            start_time = time.time()
            print(f"reduce synchronize begin*** epoch: {epoch} start time: {start_time}")
        # 
        # dist.barrier()
        for handle in self._handles:
            # 4分区2进程，一个epoch会发出20个reduce
            # 8分区2进程，一个epoch也会发出20个reduce
            # 不使用pipeline依然是20个reduce
            # 每个参数都会进行一次reduce
            # if rank == 0:
            #     print(f"reduce:{epoch}")
            handle.wait()
        self._handles.clear()
        # 注释之后，就收敛不了了
        # 因为没有等待reduce上的异步操作完成
        torch.cuda.current_stream().wait_stream(self._stream)
        if Count_Time and rank == 0:
            end_time = time.time()
            print(f"reduce synchronize end*** epoch: {epoch} end time: {end_time} cousume time: {end_time - start_time}")
