"""
Training package init.
"""
from training.curriculum import CurriculumManager
from training.eval import evaluate_agent

__all__ = ["CurriculumManager", "evaluate_agent"]
