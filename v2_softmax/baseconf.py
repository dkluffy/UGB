TARGET_SIZE = 128
TARGET_SHAPE = (TARGET_SIZE, TARGET_SIZE, 3)
MID_SIZE = 192
MID_SHAPE = (MID_SIZE,MID_SIZE,3)

FINE_TUNE_AT = 0
BASE_LR = 0.001

RESCAL_RATIO = [0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]

BASE_MODEL_FILENAME="mode_freeze_100.h5"
CLS_NUM=100