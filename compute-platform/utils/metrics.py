"""
Metrics collection utility
"""
import time
from typing import Dict, List, Any
from collections import defaultdict, deque

class MetricsCollector:
    """Simple metrics collector for tracking system performance"""
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.timers: Dict[str, List[float]] = defaultdict(list)
    
    def increment(self, metric_name: str, value: int = 1):
        """Increment a counter metric"""
        full_name = f"{self.namespace}.{metric_name}"
        self.counters[full_name] += value
    
    def gauge(self, metric_name: str, value: float):
        """Set a gauge metric"""
        full_name = f"{self.namespace}.{metric_name}"
        self.gauges[full_name] = value
    
    def histogram(self, metric_name: str, value: float):
        """Record a histogram value"""
        full_name = f"{self.namespace}.{metric_name}"
        self.histograms[full_name].append(value)
    
    def timer(self, metric_name: str):
        """Create a timer context manager"""
        return Timer(self, metric_name)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {k: list(v) for k, v in self.histograms.items()},
            'timers': dict(self.timers)
        }


class Timer:
    """Timer context manager for measuring execution time"""
    
    def __init__(self, collector: MetricsCollector, metric_name: str):
        self.collector = collector
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.histogram(self.metric_name, duration) 