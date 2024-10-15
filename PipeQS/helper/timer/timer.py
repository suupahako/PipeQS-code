from helper.timer.comm_timer import *

comm_timer = CommTimer()
true_comm_timer = TransferCommTimer(_type="full_transfer")
forward_comm_timer = TransferCommTimer(_type="half_transfer")
backward_comm_timer = TransferCommTimer(_type="half_transfer")
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

neu_timer = CommTimer()
forward_timer = CommTimer()
backward_timer = CommTimer()
