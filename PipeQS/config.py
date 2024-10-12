# config.py
FORWARD_QUANTIZATION = True
BACKWARD_QUANTIZATION = True
BITS = 2
STALE = True
GROUP_NUM = 10
ADAPTIVE = True
STALE_THRESHOLD = 2
MAX_STALE = 3
K = 10
E = 0.1
BIT_OPTIONS = [2, 4, 8 ,0]

WAIT_REDUCE = False

QT_ON_COMP = False
DEBUG = False
Count_Time = False
TRANFER_DEBUG = False
Count_Forward_Time = False
Count_Stale = False

CSV_NAME = 'training_results.csv'
WRITE_CSV = True



STALE_RESTRICT = False
STALE_RESTRICT_EPOCH = 5
STALE_CHECK_2 = False


SMOOTH = False