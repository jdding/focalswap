"""FocalSwap search-attribution audits."""

from .audit import analyze, target_rank
from .model import collect_scores, match_donors

__all__ = ["analyze", "collect_scores", "match_donors", "target_rank"]
__version__ = "0.1.0"
