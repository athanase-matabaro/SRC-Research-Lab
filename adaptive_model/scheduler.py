#!/usr/bin/env python3
"""
Compression Scheduler for Adaptive Model.
"""

import numpy as np
from collections import deque


class CompressionScheduler:
    """Dynamic scheduler for adaptive compression parameters."""

    def __init__(self, initial_threshold=0.01, decay=0.95, window_size=5):
        self.threshold = initial_threshold
        self.decay = decay
        self.window_size = window_size
        self.caq_history = deque(maxlen=window_size)
        self.threshold_history = []
        self.epoch = 0

    def update(self, caq_score):
        """Update scheduler with new CAQ score."""
        self.epoch += 1
        self.caq_history.append(caq_score)

        if len(self.caq_history) >= 2:
            recent_caqs = list(self.caq_history)
            caq_trend = recent_caqs[-1] - recent_caqs[0]

            if caq_trend > 0:
                self.threshold *= self.decay
            else:
                self.threshold /= self.decay

            self.threshold = np.clip(self.threshold, 0.001, 0.1)

        self.threshold_history.append(self.threshold)
        return self.threshold

    def get_stats(self):
        """Get scheduler statistics."""
        if len(self.caq_history) == 0:
            return {
                "epoch": self.epoch,
                "current_threshold": self.threshold,
                "mean_caq": 0.0,
                "caq_trend": 0.0
            }

        recent_caqs = list(self.caq_history)
        mean_caq = np.mean(recent_caqs)

        if len(recent_caqs) >= 2:
            caq_trend = recent_caqs[-1] - recent_caqs[0]
        else:
            caq_trend = 0.0

        return {
            "epoch": self.epoch,
            "current_threshold": self.threshold,
            "mean_caq": mean_caq,
            "caq_trend": caq_trend,
            "caq_history": recent_caqs
        }

    def reset(self):
        """Reset scheduler state."""
        self.threshold = 0.01
        self.caq_history.clear()
        self.threshold_history.clear()
        self.epoch = 0
