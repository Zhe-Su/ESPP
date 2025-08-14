from dataclasses import dataclass
import numpy as np
from typing import Callable
from abc import ABC

@dataclass
class OutputBase(ABC):
    ...


class Metric:
    def __init__(self, metric_fn:Callable) -> None:
        self._values = []
        self._buffer = []
        self.metric_fn = metric_fn

    def __getitem__(self, idx:int) -> np.ndarray:
        return self._values[idx]

    def accumulate(self, output:OutputBase) -> None:
        self._buffer.append(self.metric_fn(output))

    def compute(self) -> None:
        mean_value = np.array(self._buffer).mean() # transforming to numpy to move array from gpu
        self._buffer = []
        self._values.append(mean_value)


class MultiMetric:
    def __init__(self, tag:str, metrics:dict[str, Metric]) -> None:
        self.tag = tag
        self.metrics: dict[str, Metric] = metrics

    def __getitem__(self, key:str) -> Metric:
        return self.metrics[key]

    def accumulate(self, output:OutputBase) -> None:
        for metric in self.metrics.values():
            metric.accumulate(output)

    def compute(self) -> None:
        for metric in self.metrics.values():
            metric.compute()

    def get_mean_buffer_values(self) -> dict[str, float]:
        return_dict = {}
        for name, metric in self.metrics.items():
            return_dict[name] = sum(metric._buffer) / len(metric._buffer)
        return return_dict



class MultiMetricBuilder:
    def __init__(self):
        self.metric_fns:dict[str, Callable] = {}
        self.tag = None
    
    def add_metric(self, name: str, metric_fn:Callable) -> "MultiMetricBuilder":
        self.metric_fns[name] = metric_fn
        return self
    
    def set_tag(self, tag:str) -> "MultiMetricBuilder":
        self.tag = tag
        return self
    
    def reset(self) -> "MultiMetricBuilder":
        self.tag = None
        self.metric_fns:dict[str, Callable] = {}
        return self
    
    def build(self) -> MultiMetric:
        assert isinstance(self.tag, str)
        assert len(self.metric_fns) != 0
        
        metrics = {}
        for name, metric_fn in self.metric_fns.items():
            metrics[name] = Metric(metric_fn)
        metric_manager = MultiMetric(self.tag, metrics)
        return metric_manager