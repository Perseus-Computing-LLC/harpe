"""Modern CUDA C++ Unified Memory Emulation, Asynchronous Prefetching, and Warp Coalescing.

Derived from NVIDIA DLI: Getting Started with Accelerated Computing in Modern CUDA C++ (course-v1:DLI+S-AC-04+V2).
Provides:
- Unified Memory page pool emulation (tracking host vs device page migration and eliminating first-touch page faults via cudaMemPrefetchAsync).
- Grid-stride loop indexing utility for scalable hardware-agnostic vectorized batching.
- Warp-level memory access coalescing verification (128-byte aligned segment analysis across 32 threads).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class MemoryLocation(str, Enum):
    """Physical hardware location of a memory allocation."""
    HOST_CPU = "host_cpu"
    DEVICE_GPU = "device_gpu"


class ManagedPage(BaseModel):
    """A 4KB virtual memory page in a unified memory address space."""
    page_id: int
    size_bytes: int = 4096
    current_location: MemoryLocation = MemoryLocation.HOST_CPU
    is_dirty: bool = False
    access_count: int = 0


class UnifiedMemoryPrefetcher:
    """Emulates cudaMallocManaged and cudaMemPrefetchAsync memory lifecycle.

    Key principles:
    - Default allocation (cudaMallocManaged) starts unmapped or resident on Host.
    - Naive GPU access triggers page faults (10-40 microseconds per fault over PCIe/NVLink).
    - Explicit cudaMemPrefetchAsync stages pages to Device VRAM in advance, reducing latency to zero during kernel compute.
    """

    PAGE_SIZE: int = 4096  # Standard 4KB page

    def __init__(self, total_bytes: int, device_id: int = 0) -> None:
        if total_bytes <= 0:
            raise ValueError("total_bytes must be > 0")
        self.total_bytes = total_bytes
        self.device_id = device_id
        self.num_pages = math.ceil(total_bytes / self.PAGE_SIZE)

        # Allocate pages initially residing in Host memory
        self.pages: List[ManagedPage] = [
            ManagedPage(page_id=i, current_location=MemoryLocation.HOST_CPU)
            for i in range(self.num_pages)
        ]

        # Statistics
        self.page_fault_count = 0
        self.prefetched_page_count = 0
        self.device_hit_count = 0

    def prefetch_async(self, offset_bytes: int, count_bytes: int, destination: MemoryLocation) -> int:
        """Emulate cudaMemPrefetchAsync(ptr, count, dstDevice, stream).

        Migrates contiguous range of pages to destination without blocking compute thread.
        Returns number of pages migrated.
        """
        start_page = offset_bytes // self.PAGE_SIZE
        end_page = min(self.num_pages, math.ceil((offset_bytes + count_bytes) / self.PAGE_SIZE))

        migrated = 0
        for p_idx in range(start_page, end_page):
            page = self.pages[p_idx]
            if page.current_location != destination:
                page.current_location = destination
                migrated += 1
                self.prefetched_page_count += 1

        return migrated

    def access_page_on_device(self, offset_bytes: int) -> Dict[str, Any]:
        """Emulate memory read/write on GPU, recording page faults if page is on Host."""
        page_idx = offset_bytes // self.PAGE_SIZE
        if page_idx >= self.num_pages:
            raise IndexError(f"Memory access offset {offset_bytes} exceeds allocated size {self.total_bytes}")

        page = self.pages[page_idx]
        page.access_count += 1

        if page.current_location == MemoryLocation.HOST_CPU:
            # First-touch on-demand page fault penalty!
            self.page_fault_count += 1
            page.current_location = MemoryLocation.DEVICE_GPU
            return {
                "page_id": page_idx,
                "status": "PAGE_FAULT",
                "fault_latency_penalty_us": 25.0,  # ~25 us PCIe migration overhead
            }
        else:
            self.device_hit_count += 1
            return {
                "page_id": page_idx,
                "status": "CACHE_HIT",
                "fault_latency_penalty_us": 0.0,
            }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Summary of page faults, prefetch operations, and latency savings."""
        total_accesses = self.page_fault_count + self.device_hit_count
        fault_rate = (self.page_fault_count / total_accesses * 100.0) if total_accesses > 0 else 0.0
        # Latency estimated assuming 25 us per on-demand page fault vs 0 us for prefetched hit
        estimated_time_lost_us = self.page_fault_count * 25.0
        estimated_time_saved_us = self.prefetched_page_count * 25.0

        return {
            "total_pages": self.num_pages,
            "total_bytes": self.total_bytes,
            "page_fault_count": self.page_fault_count,
            "prefetched_page_count": self.prefetched_page_count,
            "device_hit_count": self.device_hit_count,
            "fault_rate_pct": round(fault_rate, 2),
            "estimated_penalty_us": estimated_time_lost_us,
            "estimated_savings_us": estimated_time_saved_us,
        }


class WarpCoalescingAnalyzer:
    """Verifies memory access coalescing across a 32-thread hardware warp.

    In modern CUDA architectures:
    - Global memory is loaded via 32-byte, 64-byte, or 128-byte cache lines.
    - If 32 threads in a warp access consecutive 4-byte (float32) words aligned to 128 bytes,
      the hardware fulfills the entire request in a SINGLE 128-byte memory transaction (100% bus utilization).
    - If threads access strided or random addresses, up to 32 separate transactions are serialized.
    """

    WARP_SIZE: int = 32
    TRANSACTION_SEGMENT_BYTES: int = 128

    @classmethod
    def evaluate_warp_access_pattern(
        cls,
        thread_byte_addresses: List[int],
    ) -> Dict[str, Any]:
        """Analyze memory access addresses for 32 threads in a warp.

        Calculates the number of required 128-byte hardware transactions and coalescing efficiency.
        """
        if len(thread_byte_addresses) != cls.WARP_SIZE:
            raise ValueError(f"Warp requires exactly {cls.WARP_SIZE} thread addresses (got {len(thread_byte_addresses)})")

        # Map each thread address to its 128-byte segment base
        unique_segments = set(addr // cls.TRANSACTION_SEGMENT_BYTES for addr in thread_byte_addresses)
        num_transactions = len(unique_segments)

        # Theoretical minimum transactions: 1 (if all 32 x 4-byte elements lie in one 128-byte segment)
        # Maximum transactions: 32 (if every thread hits a different segment)
        efficiency_pct = (1.0 / num_transactions) * 100.0

        is_fully_coalesced = (num_transactions == 1)

        return {
            "num_threads": cls.WARP_SIZE,
            "unique_128b_segments_accessed": num_transactions,
            "is_fully_coalesced": is_fully_coalesced,
            "coalescing_efficiency_pct": round(efficiency_pct, 2),
            "memory_bus_transaction_penalty_factor": num_transactions,
        }

    @staticmethod
    def generate_grid_stride_indices(total_items: int, grid_dim: int, block_dim: int) -> List[List[int]]:
        """Simulate thread work distribution using grid-stride loops:
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        """
        stride = grid_dim * block_dim
        thread_allocations: List[List[int]] = [[] for _ in range(stride)]

        for item_idx in range(total_items):
            tid = item_idx % stride
            thread_allocations[tid].append(item_idx)

        return thread_allocations
