"""
Orchestrator Data Models - Data classes for pipeline orchestration.

Contains:
- StageResult: Result from a single pipeline stage
- PipelineTimer: Tracks timing and token usage across pipeline stages
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    name: str
    time_ms: float
    token_usage: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "time_ms": round(self.time_ms, 2)
        }
        if self.token_usage:
            result["token_usage"] = self.token_usage
        return result


@dataclass
class PipelineTimer:
    """Tracks timing and token usage across pipeline stages."""
    stages: List[StageResult] = field(default_factory=list)
    _current_start: Optional[float] = None
    _current_name: Optional[str] = None
    
    def start(self, stage_name: str):
        """Start timing a stage."""
        self._current_name = stage_name
        self._current_start = time.perf_counter()
    
    def stop(self, token_usage: Optional[Dict[str, int]] = None):
        """Stop timing the current stage and record it."""
        if self._current_start is not None and self._current_name is not None:
            elapsed_ms = (time.perf_counter() - self._current_start) * 1000
            self.stages.append(StageResult(
                name=self._current_name,
                time_ms=elapsed_ms,
                token_usage=token_usage
            ))
            self._current_start = None
            self._current_name = None
    
    def get_total_time_ms(self) -> float:
        """Get total elapsed time across all stages."""
        return sum(s.time_ms for s in self.stages)
    
    def get_total_tokens(self) -> Dict[str, int]:
        """Get total token usage across all stages."""
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for s in self.stages:
            if s.token_usage:
                total_tokens["prompt_tokens"] += s.token_usage.get("prompt_tokens", 0)
                total_tokens["completion_tokens"] += s.token_usage.get("completion_tokens", 0)
                total_tokens["total_tokens"] += s.token_usage.get("total_tokens", 0)
        
        return total_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Return timing and token summary."""
        total_tokens = self.get_total_tokens()
        
        return {
            "stages": [s.to_dict() for s in self.stages],
            "total_time_ms": round(self.get_total_time_ms(), 2),
            "total_tokens": total_tokens if total_tokens["total_tokens"] > 0 else None
        }
