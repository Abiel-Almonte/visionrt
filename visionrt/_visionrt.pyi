"""
C++/CUDA runtime providing a GPU-resident pipeline for camera capture, preprocessing, and PyTorch integration.
"""
from __future__ import annotations
import torch
import typing
__all__: list[str] = ['Camera', 'GraphExecutor', 'fused_add_relu_cuda', 'set_verbose', 'yuyv2rgb_cuda']
class Camera:
    """
    Wrapper around a V4L2 camera device
    """
    def __init__(self, device: str) -> None:
        """
        Open a camera at the given device path (e.g., '/dev/video0').
        """
    def __iter__(self) -> Camera:
        ...
    def __next__(self) -> torch.Tensor:
        ...
    def __repr__(self) -> str:
        ...
    def close(self) -> None:
        """
        Close the camera device.
        """
    def print_formats(self) -> None:
        """
        Print all supported camera formats.
        """
    def print_selected_format(self) -> None:
        """
        Print the currently selected format.
        """
    def set_format(self, index: int) -> None:
        """
        Set the capture format by index.
        """
    def stream(self) -> Camera:
        """
        Return an iterator that yields frames.
        """
    @property
    def height(self) -> int:
        """
        Height of the current format.
        """
    @property
    def width(self) -> int:
        """
        Width of the current format.
        """
class GraphExecutor:
    """
    Modfier that compiles and captures CUDA graph for the given PyTorch module
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Launch CUDA graph
        """
    def __init__(self, module: typing.Any) -> None:
        """
        PyTorch module to compile and capture
        """
    def capture(self, tensor: torch.Tensor) -> None:
        """
        Capture CUDA graph
        """
    def is_captured(self) -> bool:
        """
        Return True if CUDA graph has been captured
        """
def fused_add_relu_cuda(arg0: torch.Tensor, arg1: torch.Tensor) -> torch.Tensor:
    """
    Fused add + relu
    """
def set_verbose(arg0: bool) -> None:
    """
    Enable/disable verbose logging
    """
def yuyv2rgb_cuda(yuyv: torch.Tensor, height: int, width: int, scale: list[float], offset: list[float]) -> torch.Tensor:
    """
    YUYV to normalized RGB CHW conversion
    """
