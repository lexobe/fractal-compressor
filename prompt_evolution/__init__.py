#!/usr/bin/env python3
"""
LLM Prompt进化算法包
实现基于遗传算法的prompt自动优化
"""

from .core import PromptIndividual, Population, TokenCounter, PromptValidator, LLMInterface
from .fitness import (
    FitnessEvaluator, 
    TextSegmentationFitnessEvaluator, 
    SimpleFitnessEvaluator,
    SegmentationSample,
    BoundaryMatcher,
    SemanticAnalyzer,
    GranularityAnalyzer
)
from .operators import (
    MutationOperator,
    CrossoverOperator, 
    SelectionOperator,
    LLMMutationOperator,
    LLMCrossoverOperator,
    EliteSelection,
    TournamentSelection,
    RouletteWheelSelection,
    EvolutionaryOperators
)
from .engine import PromptEvolutionEngine, EvolutionConfig, GenerationStats

__version__ = "0.1.0"
__author__ = "Claude Code"

__all__ = [
    # 核心类
    "PromptIndividual",
    "Population", 
    "TokenCounter",
    "PromptValidator",
    "LLMInterface",
    
    # 适应度评估
    "FitnessEvaluator",
    "TextSegmentationFitnessEvaluator",
    "SimpleFitnessEvaluator", 
    "SegmentationSample",
    "BoundaryMatcher",
    "SemanticAnalyzer",
    "GranularityAnalyzer",
    
    # 遗传操作
    "MutationOperator",
    "CrossoverOperator",
    "SelectionOperator", 
    "LLMMutationOperator",
    "LLMCrossoverOperator",
    "EliteSelection",
    "TournamentSelection", 
    "RouletteWheelSelection",
    "EvolutionaryOperators",
    
    # 进化引擎
    "PromptEvolutionEngine",
    "EvolutionConfig",
    "GenerationStats"
]