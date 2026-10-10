"""
High-Frequency Resource Profiling & Monitoring Module.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Monitors process-level and system-wide CPU utilization, physical RSS memory footprint,
and peak RSS during inference execution with high-frequency sampling (psutil).
"""

import os
import threading
import time
from typing import Dict, Any, List, Optional
import psutil


class ResourceMonitor:
    def __init__(self, sampling_interval_seconds: float = 0.05):
        self.interval = sampling_interval_seconds
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.rss_samples_mb: List[float] = []
        self.process_cpu_samples: List[float] = []
        self.system_cpu_samples: List[float] = []
        self.initial_rss_mb: float = 0.0
        self.peak_rss_mb: float = 0.0

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                # Memory measurement
                mem_info = self.process.memory_info()
                rss_mb = round(mem_info.rss / (1024 * 1024), 2)
                self.rss_samples_mb.append(rss_mb)
                if rss_mb > self.peak_rss_mb:
                    self.peak_rss_mb = rss_mb

                # CPU measurements
                p_cpu = self.process.cpu_percent(interval=None)
                s_cpu = psutil.cpu_percent(interval=None)
                self.process_cpu_samples.append(p_cpu)
                self.system_cpu_samples.append(s_cpu)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)

    def start(self):
        """Starts background resource monitoring thread."""
        self._stop_event.clear()
        self.rss_samples_mb = []
        self.process_cpu_samples = []
        self.system_cpu_samples = []

        # Initial baseline sample
        mem_info = self.process.memory_info()
        self.initial_rss_mb = round(mem_info.rss / (1024 * 1024), 2)
        self.peak_rss_mb = self.initial_rss_mb
        self.process.cpu_percent(interval=None) # Prime CPU counter

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stops monitoring and returns statistical profile."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.0)
            self._thread = None

        # Final sample
        try:
            mem_info = self.process.memory_info()
            final_rss = round(mem_info.rss / (1024 * 1024), 2)
            if final_rss > self.peak_rss_mb:
                self.peak_rss_mb = final_rss
        except Exception:
            final_rss = self.peak_rss_mb

        avg_p_cpu = (
            round(sum(self.process_cpu_samples) / len(self.process_cpu_samples), 2)
            if self.process_cpu_samples else 0.0
        )
        peak_p_cpu = (
            round(max(self.process_cpu_samples), 2)
            if self.process_cpu_samples else 0.0
        )
        avg_s_cpu = (
            round(sum(self.system_cpu_samples) / len(self.system_cpu_samples), 2)
            if self.system_cpu_samples else 0.0
        )

        return {
            "initial_rss_mb": self.initial_rss_mb,
            "final_rss_mb": final_rss,
            "peak_rss_mb": self.peak_rss_mb,
            "net_rss_delta_mb": round(self.peak_rss_mb - self.initial_rss_mb, 2),
            "avg_process_cpu_percent": avg_p_cpu,
            "peak_process_cpu_percent": peak_p_cpu,
            "avg_system_cpu_percent": avg_s_cpu,
            "samples_count": len(self.rss_samples_mb)
        }


if __name__ == "__main__":
    import numpy as np

    monitor = ResourceMonitor(sampling_interval_seconds=0.02)
    print("Testing Resource Monitor...")
    monitor.start()

    # Simulate memory allocation and computation
    time.sleep(0.1)
    a = np.random.randn(2000, 2000).astype(np.float64)
    b = np.dot(a, a)
    time.sleep(0.1)

    profile = monitor.stop()
    print("Profile Results:", profile)
