"""
Index Handler - Applies positional indexing to node lists.
"""

import xml.etree.ElementTree as ET
from typing import List, Union

from pipeline_execution.semantic_xpath_parsing.parsing_models import Index, IndexRange


class IndexHandler:
    """
    Handles positional index operations on node lists.
    
    Supports:
    - Single positive index: [1], [2] (1-based)
    - Negative index: [-1] (last element)
    - Range index: [1:3] (inclusive on both ends)
    - Range to end: [-2:] (last N elements)
    """
    
    @staticmethod
    def apply_index(
        nodes: List[ET.Element],
        index: Union[IndexRange, Index],
        execution_log: List[str] = None
    ) -> List[ET.Element]:
        """
        Apply positional index or range to nodes.
        
        Args:
            nodes: List of XML elements to select from
            index: IndexRange specifying position(s) to select
            execution_log: Optional list to append log messages
            
        Returns:
            List of selected nodes
        """
        if execution_log is None:
            execution_log = []
            
        if not nodes:
            return []
        
        if index.to_end:
            return IndexHandler._apply_to_end_index(nodes, index, execution_log)
        elif index.is_range:
            return IndexHandler._apply_range_index(nodes, index, execution_log)
        else:
            return IndexHandler._apply_single_index(nodes, index, execution_log)
    
    @staticmethod
    def _apply_to_end_index(
        nodes: List[ET.Element],
        index: Union[IndexRange, Index],
        execution_log: List[str]
    ) -> List[ET.Element]:
        """Apply 'to end' indexing like [-2:] (from position to end)."""
        n = len(nodes)
        start_val = index.start
        
        if start_val < 0:
            # Negative index: -2 means "from 2nd to last" = last 2 elements
            # Python: nodes[-2:] gives last 2
            start_idx = n + start_val
            start_idx = max(0, start_idx)  # Clamp to 0
        else:
            # Positive index: 3 means "from 3rd to end" (1-based)
            start_idx = start_val - 1
        
        if start_idx < n:
            result = nodes[start_idx:]
            execution_log.append(
                f"Selected nodes at range [{start_val}:] "
                f"(0-based: {start_idx}:end, got {len(result)} nodes)"
            )
            return result
        else:
            execution_log.append(
                f"Range [{start_val}:] out of range (have {n} nodes)"
            )
            return []
    
    @staticmethod
    def _apply_range_index(
        nodes: List[ET.Element],
        index: Union[IndexRange, Index],
        execution_log: List[str]
    ) -> List[ET.Element]:
        """Apply range indexing [start:end] (both 1-based, inclusive)."""
        start_idx = index.start - 1  # Convert to 0-based
        end_idx = index.end  # end is inclusive, so we use it directly for slice
        
        # Clamp to valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(nodes), end_idx)
        
        if start_idx < len(nodes):
            result = nodes[start_idx:end_idx]
            execution_log.append(
                f"Selected nodes at range [{index.start}:{index.end}] "
                f"(0-based: {start_idx}:{end_idx}, got {len(result)} nodes)"
            )
            return result
        else:
            execution_log.append(
                f"Range [{index.start}:{index.end}] out of range "
                f"(have {len(nodes)} nodes)"
            )
            return []
    
    @staticmethod
    def _apply_single_index(
        nodes: List[ET.Element],
        index: Union[IndexRange, Index],
        execution_log: List[str]
    ) -> List[ET.Element]:
        """Apply single index (1-based positive, or negative from end)."""
        idx_val = index.start
        n = len(nodes)
        
        if idx_val > 0:
            # Positive: 1-based (1 = first element)
            idx = idx_val - 1
        elif idx_val < 0:
            # Negative: from end (-1 = last, -2 = second from last)
            idx = n + idx_val
        else:
            # Zero index is invalid in 1-based system
            execution_log.append(f"Invalid index 0 (indices are 1-based)")
            return []
        
        if 0 <= idx < n:
            execution_log.append(f"Selected node at index {idx_val} (0-based: {idx})")
            return [nodes[idx]]
        else:
            execution_log.append(
                f"Index {idx_val} out of range (have {n} nodes)"
            )
            return []


# Convenience function
apply_index = IndexHandler.apply_index
