import React, { useState, useEffect, useCallback } from "react";
import {
  FolderOpen,
  Folder,
  FileCode2,
  Loader2,
  AlertCircle,
  Database,
  FlaskConical,
  ChevronRight,
} from "lucide-react";

interface FileNode {
  name: string;
  relativePath: string;
  type: "file" | "directory";
  children?: FileNode[];
}

interface FileBrowserProps {
  onXmlLoaded: (content: string) => void;
}

const TreeNode: React.FC<{
  node: FileNode;
  depth: number;
  onSelect: (path: string) => void;
  loadingPath: string | null;
  selectedPath: string | null;
}> = ({ node, depth, onSelect, loadingPath, selectedPath }) => {
  const [expanded, setExpanded] = useState(depth < 2);

  if (node.type === "directory") {
    const hasChildren = node.children && node.children.length > 0;
    return (
      <div className="tree-browser-node">
        <button
          className={`tree-browser-row tree-browser-folder ${expanded ? "expanded" : ""}`}
          onClick={() => setExpanded(!expanded)}
          style={{ paddingLeft: `${depth * 1.25 + 0.75}rem` }}
        >
          <ChevronRight
            size={14}
            className={`tree-browser-chevron ${expanded ? "rotated" : ""}`}
          />
          {expanded ? (
            <FolderOpen size={16} className="tree-browser-icon folder" />
          ) : (
            <Folder size={16} className="tree-browser-icon folder" />
          )}
          <span className="tree-browser-label">{node.name}</span>
          {hasChildren && (
            <span className="tree-browser-count">
              {countXmlFiles(node)} files
            </span>
          )}
        </button>
        {expanded && hasChildren && (
          <div className="tree-browser-children">
            {node.children!.map((child) => (
              <TreeNode
                key={child.relativePath}
                node={child}
                depth={depth + 1}
                onSelect={onSelect}
                loadingPath={loadingPath}
                selectedPath={selectedPath}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  const isLoading = loadingPath === node.relativePath;
  const isSelected = selectedPath === node.relativePath;

  return (
    <button
      className={`tree-browser-row tree-browser-file ${isSelected ? "selected" : ""} ${isLoading ? "loading" : ""}`}
      style={{ paddingLeft: `${depth * 1.25 + 0.75}rem` }}
      onClick={() => onSelect(node.relativePath)}
      disabled={isLoading}
    >
      <span className="tree-browser-spacer" />
      {isLoading ? (
        <Loader2 size={15} className="tree-browser-icon file spinning" />
      ) : (
        <FileCode2 size={15} className="tree-browser-icon file" />
      )}
      <span className="tree-browser-label">{node.name}</span>
    </button>
  );
};

function countXmlFiles(node: FileNode): number {
  if (node.type === "file") return 1;
  if (!node.children) return 0;
  return node.children.reduce((sum, child) => sum + countXmlFiles(child), 0);
}

export const FileBrowser: React.FC<FileBrowserProps> = ({ onXmlLoaded }) => {
  const [tree, setTree] = useState<FileNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/xml-files")
      .then((res) => res.json())
      .then((data) => {
        setTree(data);
        setLoading(false);
      })
      .catch((err) => {
        setError("Failed to load file listing: " + err.message);
        setLoading(false);
      });
  }, []);

  const handleSelect = useCallback(
    async (relativePath: string) => {
      setLoadingPath(relativePath);
      setSelectedPath(relativePath);
      try {
        const res = await fetch(
          `/api/xml-content?path=${encodeURIComponent(relativePath)}`,
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const content = await res.text();
        onXmlLoaded(content);
      } catch (err: any) {
        setError("Failed to load file: " + err.message);
      } finally {
        setLoadingPath(null);
      }
    },
    [onXmlLoaded],
  );

  // Assign icons to root-level groups
  const rootIcons: Record<string, React.ReactNode> = {
    "Storage / Memory": <Database size={18} />,
    Experiments: <FlaskConical size={18} />,
  };

  if (loading) {
    return (
      <div className="file-browser-container">
        <div className="file-browser-loading">
          <Loader2 size={24} className="spinning" />
          <span>Scanning project for XML files…</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="file-browser-container">
        <div className="file-browser-error">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="file-browser-container">
      <div className="file-browser-header">
        <FolderOpen size={18} />
        <span>Project XML Files</span>
      </div>
      <div className="file-browser-tree">
        {tree.map((rootNode) => (
          <div key={rootNode.relativePath} className="file-browser-group">
            <div className="file-browser-group-header">
              {rootIcons[rootNode.name] || <Folder size={18} />}
              <span>{rootNode.name}</span>
              <span className="tree-browser-count">
                {countXmlFiles(rootNode)} files
              </span>
            </div>
            <div className="file-browser-group-content">
              {rootNode.children && rootNode.children.length > 0 ? (
                rootNode.children.map((child) => (
                  <TreeNode
                    key={child.relativePath}
                    node={child}
                    depth={1}
                    onSelect={handleSelect}
                    loadingPath={loadingPath}
                    selectedPath={selectedPath}
                  />
                ))
              ) : (
                <div className="file-browser-empty">No XML files found</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
