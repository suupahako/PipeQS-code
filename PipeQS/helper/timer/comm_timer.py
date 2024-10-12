import time
import torch.distributed as dist
from contextlib import contextmanager
from config import *

class CommTimer(object):

    def __init__(self):
        super(CommTimer, self).__init__()
        self._time = {}

    # 每个epoch结束后,都会进行timer的clear,因此计算的就是一个epoch的通讯等待时间
    @contextmanager
    def timer(self, name):
        if name in self._time:
            raise Exception(name + " already exists")
        t0 = time.time()
        yield
        t1 = time.time()
        self._time[name] = (t0, t1)

    def tot_time(self):
        tot = 0
        for (t0, t1) in self._time.values():
            tot += t1 - t0
        return tot

    def print_time(self):
        rank, size = dist.get_rank(), dist.get_world_size()
        for (k, (t0, t1)) in self._time.items():
            print(f'(rank {rank}) Communication time of {k}: {t1 - t0} seconds.')

    def clear(self):
        self._time = {}


# class TransferCommTimer(object):

#     def __init__(self, _type):
#         super(TransferCommTimer, self).__init__()
#         self._num = {}
#         self._tot = 0
#         # self._cnt = 0
#         self._type = _type
#         # 先不管了，在train里面直接传
#         self._size = 1
        

#     # 每个epoch结束后,都会进行timer的clear,因此计算的就是一个epoch的通讯等待时间
#     @contextmanager
#     def timer(self, name):
#         # if self._type == "half_for" or self._type == "full_for":
#         #     if name not in self._qt_num:
#         #         self._qt_num[name] = 1
#         #     else:
#         #         self._qt_num[name] += 1
#         if name in self._num:
#             raise Exception(name + " already exists")
#         t0 = time.time()
#         yield
#         t1 = time.time()
#         self._num[name] = 1
#         # self._cnt += 1
#         self._tot += (t1 - t0)


#     def avg_time(self, world_size):
#         self._size = world_size
#         cnt = 0
#         for num in self._num.values():
#             cnt += num
#         print(cnt)
#         if cnt == 0:
#             cnt = 1
#         # if self._cnt == 0:
#         #     self._cnt = 1
#         # if self._type == "half_for" or self._type == "full_for":
#         #     for num in self._qt_num.values():
#         #         # 不知道为什么打出来是2
#         #         print(f"***{num}")
#         #         tot_num = num 
#         # print(self._cnt)
#         if self._type == "full_transfer":
#             # return (self._tot / self._cnt * 6)
#             return (self._tot / cnt * 6)
#         elif self._type == "half_transfer":
#             # return (self._tot / self._cnt * 3)
#             return (self._tot / cnt * 3)
#         elif self._type == "half_for":
#             # return (self._tot / self._cnt * 3 * self._size)
#             return (self._tot / cnt * 3 * self._size)
#         else:
#             # return (self._tot / self._cnt * 6 * self._size)
#             return (self._tot / cnt * 6 * self._size)
        
        

#     def print_time(self):
#         rank, size = dist.get_rank(), dist.get_world_size()
#         for (k, (t0, t1)) in self._time.items():
#             print(f'(rank {rank}) Communication time of {k}: {t1 - t0} seconds.')

#     def clear(self):
#         self._num = {}
#         self._tot = 0
#         self._cnt = 0
#         self._type = _type

class TransferCommTimer(object):

    def __init__(self, _type):
        super(TransferCommTimer, self).__init__()
        self._num = {}
        self._time = {}
        self._tot = 0
        self._cnt = 0
        self._type = _type
        # 先不管了，在train里面直接传
        self._size = 1

    @contextmanager
    def timer(self, name):
        # if self._type == "half_for" or self._type == "full_for":
        #     if name not in self._qt_num:
        #         self._qt_num[name] = 1
        #     else:
        #         self._qt_num[name] += 1
        if name in self._num:
            raise Exception(name + " already exists")
        t0 = time.time()
        yield
        t1 = time.time()
        self._num[name] = 1
        self._time[name] = (t0, t1)
        # self._cnt += 1
        # self._tot += (t1 - t0)

    def tot_time(self):
        for (t0, t1) in self._time.values():
           self._tot += t1 - t0
        self._time = {}


    def get_transfer_num(self):
        for num in self._num.values():
            self._cnt += num
        self._num = {}
        return self._cnt

    def avg_time(self, world_size):
        self._size = world_size
        self.get_transfer_num()
        if self._cnt == 0:
            return 0

        self.tot_time()
        # if self._cnt == 0:
        #     self._cnt = 1
        # if self._type == "half_for" or self._type == "full_for":
        #     for num in self._qt_num.values():
        #         # 不知道为什么打出来是2
        #         print(f"***{num}")
        #         tot_num = num 
        # print(self._cnt)
        if self._type == "full_transfer":
            # return (self._tot / self._cnt * 6)
            return (self._tot / self._cnt * 6)
        elif self._type == "half_transfer":
            # return (self._tot / self._cnt * 3)
            return (self._tot / self._cnt * 3)
        elif self._type == "half_for":
            # return (self._tot / self._cnt * 3 * self._size)
            return (self._tot / self._cnt * 3 * self._size)
        else:
            # return (self._tot / self._cnt * 6 * self._size)
            return (self._tot / self._cnt * 6 * self._size)
        


    def print_time(self):
        rank, size = dist.get_rank(), dist.get_world_size()
        for (k, (t0, t1)) in self._time.items():
            print(f'(rank {rank}) Communication time of {k}: {t1 - t0} seconds.')

    def clear(self):
        self._qt_num = {}
        self._tot = 0
        self._cnt = 0
        self._type = _type


        