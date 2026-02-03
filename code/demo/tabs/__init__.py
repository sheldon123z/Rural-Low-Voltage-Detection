"""
Gradio Demo 标签页模块
"""

from .tab1_principle import create_principle_tab
from .tab2_innovation import create_innovation_tab
from .tab3_arena import create_arena_tab
from .tab4_detection import create_detection_tab

__all__ = [
    "create_principle_tab",
    "create_innovation_tab",
    "create_arena_tab",
    "create_detection_tab",
]
