import React, { useState } from "react";
import { ChevronRight, ChevronDown, FileJson, Tag } from "lucide-react";

interface GenericViewProps {
  data: any;
}

const TreeNode: React.FC<{ label: string; value: any; level?: number }> = ({
  label,
  value,
  level = 0,
}) => {
  const [expanded, setExpanded] = useState(true);
  const isObject = value !== null && typeof value === "object";
  const isArray = Array.isArray(value);

  const toggleExpand = (e: React.MouseEvent) => {
    e.stopPropagation();
    setExpanded(!expanded);
  };

  if (!isObject) {
    return (
      <div className="tree-node" style={{ paddingLeft: `${level * 20}px` }}>
        <div className="node-content leaf">
          <span className="node-key">{label}:</span>
          <span className="node-value">{String(value)}</span>
        </div>
      </div>
    );
  }

  // Handle object or array
  const keys = Object.keys(value);
  const isEmpty = keys.length === 0;

  return (
    <div className="tree-node-group">
      <div
        className="tree-node parent"
        onClick={toggleExpand}
        style={{ paddingLeft: `${level * 20}px` }}
      >
        <div className="node-content">
          {isEmpty ? (
            <span className="spacer" />
          ) : expanded ? (
            <ChevronDown size={14} />
          ) : (
            <ChevronRight size={14} />
          )}
          <Tag size={14} className="node-icon" />
          <span className="node-key">{label}</span>
          {isArray && <span className="node-meta">[{value.length}]</span>}
        </div>
      </div>

      {expanded && !isEmpty && (
        <div className="node-children">
          {keys.map((key) => (
            <TreeNode
              key={key}
              label={key}
              value={value[key as keyof typeof value]}
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export const GenericView: React.FC<GenericViewProps> = ({ data }) => {
  return (
    <div className="generic-view">
      <h1 className="view-title">
        <FileJson size={24} />
        XML Content Viewer
      </h1>
      <div className="tree-container">
        {Object.keys(data).map((key) => (
          <TreeNode key={key} label={key} value={data[key]} />
        ))}
      </div>
    </div>
  );
};
