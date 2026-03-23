"""
Pipeline Summary Logger - Formats and prints pipeline execution summaries.

Provides formatted console output for:
- Stage-by-stage timing breakdown
- Token usage per stage
- Progress bar visualization
- Total execution statistics
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline_execution.pipeline_orchestrator.orchestrator_models import PipelineTimer


class PipelineSummaryLogger:
    """
    Formats and prints pipeline execution summaries.
    
    Provides visual progress bars and token usage breakdowns
    for each stage of pipeline execution.
    """
    
    @staticmethod
    def print_summary(timer: "PipelineTimer"):
        """
        Print a formatted timing and token summary.
        
        Args:
            timer: PipelineTimer instance with recorded stages
        """
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        
        total = timer.get_total_time_ms()
        total_prompt = 0
        total_completion = 0
        
        for stage in timer.stages:
            pct = (stage.time_ms / total * 100) if total > 0 else 0
            bar_len = int(pct / 5)
            bar = "#" * bar_len + "." * (20 - bar_len)
            
            # Format tokens if available
            token_str = ""
            if stage.token_usage:
                prompt = stage.token_usage.get("prompt_tokens", 0)
                completion = stage.token_usage.get("completion_tokens", 0)
                total_prompt += prompt
                total_completion += completion
                token_str = f" [{prompt}+{completion} tokens]"
            
            print(f"  {stage.name:<30} {stage.time_ms:>8.1f}ms  {bar} {pct:>5.1f}%{token_str}")
        
        print("-" * 70)
        print(f"  {'TOTAL':<30} {total:>8.1f}ms")
        if total_prompt > 0 or total_completion > 0:
            print(f"  {'TOKENS':<30} {total_prompt} prompt + {total_completion} completion = {total_prompt + total_completion} total")
        print("=" * 70)
    
    @staticmethod
    def format_stage_line(
        stage_name: str,
        time_ms: float,
        total_ms: float,
        token_usage: dict = None
    ) -> str:
        """
        Format a single stage line for summary output.
        
        Args:
            stage_name: Name of the stage
            time_ms: Time taken for this stage in milliseconds
            total_ms: Total pipeline time for percentage calculation
            token_usage: Optional token usage dict
            
        Returns:
            Formatted string for the stage line
        """
        pct = (time_ms / total_ms * 100) if total_ms > 0 else 0
        bar_len = int(pct / 5)
        bar = "#" * bar_len + "." * (20 - bar_len)
        
        token_str = ""
        if token_usage:
            prompt = token_usage.get("prompt_tokens", 0)
            completion = token_usage.get("completion_tokens", 0)
            token_str = f" [{prompt}+{completion} tokens]"
        
        return f"  {stage_name:<30} {time_ms:>8.1f}ms  {bar} {pct:>5.1f}%{token_str}"
    
    @staticmethod
    def log_stage_start(stage_name: str):
        """Log the start of a pipeline stage."""
        print(f"\n[Starting] {stage_name}")
    
    @staticmethod
    def log_stage_complete(stage_name: str, time_ms: float):
        """Log the completion of a pipeline stage."""
        print(f"[Done] {stage_name} ({time_ms:.1f}ms)")
