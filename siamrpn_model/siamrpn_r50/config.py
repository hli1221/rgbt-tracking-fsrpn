class BACKBONE:
    # BACKBONE
    LAYERS = [2, 3, 4]
    CHANNELS = [512, 1024, 2048]


class ADJUST:
    ADJUST_CHANNEL = [256, 256, 256]


class RPN:
    WEIGHTED = True


class ANCHOR:
    STRIDE = 8
    RATIOS = [0.33, 0.5, 1, 2, 3]
    SCALES = [8]
    ANCHOR_NUM = 5


class TRACK:
    PENALTY_K = 0.05
    WINDOW_INFLUENCE = 0.42
    LR = 0.38
    EXEMPLAR_SIZE = 127
    INSTANCE_SIZE = 255
    BASE_SIZE = 8
    CONTEXT_AMOUNT = 0.5