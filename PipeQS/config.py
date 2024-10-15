# Quantization parameters
FORWARD_QUANTIZATION = True
BACKWARD_QUANTIZATION = True
BITS = 2

# Staleness parameters
STALE = False
STALE_THRESHOLD = 2

# Adaptive bit-width and staleness adjustment parameters
ADAPTIVE = True
K = 10
E = 0.1
BIT_OPTIONS = [2, 4, 8 ,0]

# CSV parameters
CSV_NAME = 'training_results.csv'
WRITE_CSV = True

# Other parameters
GROUP_NUM = 10
QT_ON_COMP = False


