import numpy as np
from scipy.signal import savgol_filter

# 基类
class ActionFilter:
    def apply(self, action_seq: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# 滑动平均滤波器
class MovingAverageFilter(ActionFilter):
    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def apply(self, action_seq: np.ndarray) -> np.ndarray:
        recent = action_seq[-16:]  # (16, D)
        smoothed = []
        for d in range(recent.shape[1]):
            series = recent[:, d]
            padded = np.pad(series, (self.window_size - 1, 0), mode='edge')
            kernel = np.ones(self.window_size) / self.window_size
            smooth = np.convolve(padded, kernel, mode='valid')
            smoothed.append(smooth)
        return np.stack(smoothed, axis=1)

# 指数滑动平均滤波器
class ExponentialMovingAverageFilter(ActionFilter):
    def __init__(self, alpha: float = 0.2, length = 16):
        self.alpha = alpha
        self.length = length
    def apply(self, action_seq: np.ndarray) -> np.ndarray:
        if action_seq.shape[0] < self.length:
            recent = action_seq
            length = action_seq.shape[0]
        else:
            recent = action_seq[-self.length:]
            length = self.length
        smoothed = np.zeros_like(recent)
        smoothed[0] = recent[0]
        for t in range(1, length):
            smoothed[t] = self.alpha * recent[t] + (1 - self.alpha) * smoothed[t - 1]
        return smoothed

# Savitzky-Golay滤波器
class SavitzkyGolayFilter(ActionFilter):
    def __init__(self, window_length: int = 5, polyorder: int = 2):
        self.window_length = window_length
        self.polyorder = polyorder

    def apply(self, action_seq: np.ndarray) -> np.ndarray:
        recent = action_seq[-16:]
        smoothed = savgol_filter(recent, window_length=self.window_length, polyorder=self.polyorder, axis=0, mode='interp')
        return smoothed
