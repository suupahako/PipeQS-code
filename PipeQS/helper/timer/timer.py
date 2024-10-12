from helper.timer.comm_timer import *

# 用于计算非流水模式的通讯时间，或流水模式同一层通讯和通讯之间的等待时间
comm_timer = CommTimer()
# 用于计算通讯真正时间
true_comm_timer = TransferCommTimer(_type="full_transfer")
# 用于计算前向传播通讯代码花费的时间
forward_comm_timer = TransferCommTimer(_type="half_transfer")
# 用于计算反向传播通讯代码花费的时间
backward_comm_timer = TransferCommTimer(_type="half_transfer")
# 用于计算量化花费的时间
# 用于计算解量化花费的时间
# 用于计算组装tensor花费的时间
# 用于计算解包tensor花费的时间
if BACKWARD_QUANTIZATION:
    qt_timer = TransferCommTimer(_type="full_transfer")
    dq_timer = TransferCommTimer(_type="full_for")
    pack_timer = TransferCommTimer(_type="full_for")
    unpack_timer = TransferCommTimer(_type="full_for")
else:
    qt_timer = TransferCommTimer(_type="half_transfer")
    dq_timer = TransferCommTimer(_type="half_for")
    pack_timer = TransferCommTimer(_type="half_for")
    unpack_timer = TransferCommTimer(_type="half_for")


# 用于计算神经网络花费的时间
neu_timer = CommTimer()
# 用于计算前向传播花费的时间
forward_timer = CommTimer()
# 用于计算反向传播花费的时间
backward_timer = CommTimer()
