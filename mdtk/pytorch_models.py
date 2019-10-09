import torch.nn as nn


class ErrorDetectionNet(nn.Module):
    """
    Baseline model for the Error Detection task, in which the label for each
    data point is either 1 (degraded) or 0 (not degraded).
    """
    def __init__(self):
        super().__init__()
        


class ErrorClassificationNet(nn.Module):
    """
    Baseline model for the Error Classification task, in which the label for
    each data point is a degradation_id (with 0 = not degraded).
    """
    def __init__(self):
        super().__init__()



class ErrorIdentificationNet(nn.Module):
    """
    Baseline model for the Error Identification task, in which the label for
    each data point is a binary label for each frame of input, with  0 = not
    degraded and 1 = degraded.
    """
    def __init__(self):
        super().__init__()



class ErrorCorrectionNet(nn.Module):
    """
    Baseline model for the Error Correction task, in which the label for each
    data point is the clean data.
    """
    def __init__(self):
        super().__init__()
