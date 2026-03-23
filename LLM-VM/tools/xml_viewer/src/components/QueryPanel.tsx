import React, { useState, useEffect, useCallback } from "react";
import yaml from "js-yaml";
import {
  FileText,
  ChevronDown,
  Search,
  BookOpen,
  PlusCircle,
  Pencil,
  Trash2,
  HelpCircle,
  Loader2,
  AlertCircle,
  X,
  MessageSquare,
} from "lucide-react";

interface QueryFile {
  name: string;
  relativePath: string;
}

// A single query can be a string (single-turn) or an array of strings (multi-turn)
type QueryEntry = string | string[];

interface QueryGroup {
  name: string;
  use_versions: string;
  queries: QueryEntry[];
  ground_truth?: string[][];
}

interface ParsedQueryFile {
  name: string;
  groups: QueryGroup[];
}

const QUERY_TYPE_META: Record<
  string,
  { icon: React.ReactNode; color: string; bg: string; border: string }
> = {
  READ: {
    icon: <Search size={14} />,
    color: "#0369a1",
    bg: "#f0f9ff",
    border: "#bae6fd",
  },
  read_queries: {
    icon: <Search size={14} />,
    color: "#0369a1",
    bg: "#f0f9ff",
    border: "#bae6fd",
  },
  CREATE: {
    icon: <PlusCircle size={14} />,
    color: "#166534",
    bg: "#f0fdf4",
    border: "#bbf7d0",
  },
  create_queries: {
    icon: <PlusCircle size={14} />,
    color: "#166534",
    bg: "#f0fdf4",
    border: "#bbf7d0",
  },
  UPDATE: {
    icon: <Pencil size={14} />,
    color: "#854d0e",
    bg: "#fefce8",
    border: "#fde68a",
  },
  update_queries: {
    icon: <Pencil size={14} />,
    color: "#854d0e",
    bg: "#fefce8",
    border: "#fde68a",
  },
  DELETE: {
    icon: <Trash2 size={14} />,
    color: "#991b1b",
    bg: "#fef2f2",
    border: "#fecaca",
  },
  delete_queries: {
    icon: <Trash2 size={14} />,
    color: "#991b1b",
    bg: "#fef2f2",
    border: "#fecaca",
  },
  "multi-turn": {
    icon: <MessageSquare size={14} />,
    color: "#6d28d9",
    bg: "#f5f3ff",
    border: "#c4b5fd",
  },
  multi_turn: {
    icon: <MessageSquare size={14} />,
    color: "#6d28d9",
    bg: "#f5f3ff",
    border: "#c4b5fd",
  },
};

// Operation type detection helpers for multi-turn turn labels
const OPERATION_KEYWORDS: Record<string, { label: string; color: string }> = {
  cancel: { label: "DELETE", color: "#991b1b" },
  remove: { label: "DELETE", color: "#991b1b" },
  delete: { label: "DELETE", color: "#991b1b" },
  skip: { label: "DELETE", color: "#991b1b" },
  add: { label: "CREATE", color: "#166534" },
  "add a": { label: "CREATE", color: "#166534" },
  update: { label: "UPDATE", color: "#854d0e" },
  change: { label: "UPDATE", color: "#854d0e" },
  replace: { label: "UPDATE", color: "#854d0e" },
  pushed: { label: "UPDATE", color: "#854d0e" },
  "what ": { label: "READ", color: "#0369a1" },
  "which ": { label: "READ", color: "#0369a1" },
  "show ": { label: "READ", color: "#0369a1" },
  "list ": { label: "READ", color: "#0369a1" },
};

function detectOperation(query: string): { label: string; color: string } | null {
  const lower = query.toLowerCase();
  for (const [keyword, meta] of Object.entries(OPERATION_KEYWORDS)) {
    if (lower.includes(keyword)) {
      return meta;
    }
  }
  return null;
}

function getTypeMeta(name: string) {
  return (
    QUERY_TYPE_META[name] || {
      icon: <HelpCircle size={14} />,
      color: "#475569",
      bg: "#f8fafc",
      border: "#e2e8f0",
    }
  );
}

function formatTypeName(name: string): string {
  // "read_queries" → "Read", "CREATE" → "Create", "multi-turn" → "Multi Turn"
  return name
    .replace(/_queries$/i, "")
    .replace(/^./, (c) => c.toUpperCase())
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Count total individual queries (multi-turn arrays count each turn) */
function countQueries(queries: QueryEntry[]): { total: number; conversations: number; singles: number } {
  let total = 0;
  let conversations = 0;
  let singles = 0;
  for (const q of queries) {
    if (Array.isArray(q)) {
      conversations++;
      total += q.length;
    } else {
      singles++;
      total++;
    }
  }
  return { total, conversations, singles };
}

export const QueryPanel: React.FC = () => {
  const [files, setFiles] = useState<QueryFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [parsedData, setParsedData] = useState<ParsedQueryFile | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [expandedConversations, setExpandedConversations] = useState<Set<string>>(new Set());
  const [isOpen, setIsOpen] = useState(false);

  // Fetch list of query files
  useEffect(() => {
    fetch("/api/query-files")
      .then((res) => res.json())
      .then((data: QueryFile[]) => {
        setFiles(data);
        setLoading(false);
      })
      .catch((err) => {
        setError("Failed to load query files: " + err.message);
        setLoading(false);
      });
  }, []);

  const handleSelectFile = useCallback(async (relativePath: string) => {
    setSelectedFile(relativePath);
    setLoadingContent(true);
    setError(null);

    try {
      const res = await fetch(
        `/api/query-content?path=${encodeURIComponent(relativePath)}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const content = await res.text();

      // Parse YAML
      const parsed = yaml.load(content) as any;
      if (!parsed || !parsed.queries) {
        throw new Error("Invalid query file format");
      }

      const groups: QueryGroup[] = parsed.queries.map((group: any) => ({
        name: group.name || "Unnamed",
        use_versions: group.use_versions || "false",
        queries: Array.isArray(group.queries) ? group.queries : [],
        ground_truth: group.ground_truth || undefined,
      }));

      setParsedData({
        name: parsed.name || relativePath.split("/").pop() || "Unknown",
        groups,
      });

      // Auto-expand all groups
      setExpandedGroups(new Set(groups.map((g) => g.name)));
      // Auto-expand all conversations
      const convKeys = new Set<string>();
      groups.forEach((g) => {
        g.queries.forEach((q, idx) => {
          if (Array.isArray(q)) {
            convKeys.add(`${g.name}-${idx}`);
          }
        });
      });
      setExpandedConversations(convKeys);
    } catch (err: any) {
      setError("Failed to parse query file: " + err.message);
      setParsedData(null);
    } finally {
      setLoadingContent(false);
    }
  }, []);

  const toggleGroup = (name: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  const toggleConversation = (key: string) => {
    setExpandedConversations((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  return (
    <div className={`query-panel ${isOpen ? "open" : ""}`}>
      {/* Toggle tab on the side */}
      <button
        className="query-panel-toggle"
        onClick={() => setIsOpen(!isOpen)}
        title={isOpen ? "Close query panel" : "Open query panel"}
      >
        <BookOpen size={16} />
        <span>Queries</span>
      </button>

      {isOpen && (
        <div className="query-panel-content">
          <div className="query-panel-header">
            <div className="query-panel-header-left">
              <BookOpen size={18} />
              <span>Experiment Queries</span>
            </div>
            <button
              className="query-panel-close"
              onClick={() => setIsOpen(false)}
            >
              <X size={16} />
            </button>
          </div>

          {/* File Selector */}
          <div className="query-file-selector">
            {loading ? (
              <div className="query-panel-loading">
                <Loader2 size={16} className="spinning" />
                <span>Loading…</span>
              </div>
            ) : (
              <select
                className="query-file-dropdown"
                value={selectedFile || ""}
                onChange={(e) => {
                  if (e.target.value) handleSelectFile(e.target.value);
                }}
              >
                <option value="">Select a query file…</option>
                {files.map((f) => (
                  <option key={f.relativePath} value={f.relativePath}>
                    {f.name}
                  </option>
                ))}
              </select>
            )}
          </div>

          {error && (
            <div className="query-panel-error">
              <AlertCircle size={14} />
              <span>{error}</span>
            </div>
          )}

          {loadingContent && (
            <div className="query-panel-loading">
              <Loader2 size={16} className="spinning" />
              <span>Loading queries…</span>
            </div>
          )}

          {/* Query Groups */}
          {parsedData && !loadingContent && (
            <div className="query-groups">
              <div className="query-file-name">
                <FileText size={14} />
                <span>{parsedData.name}</span>
              </div>

              {parsedData.groups.map((group) => {
                const meta = getTypeMeta(group.name);
                const isExpanded = expandedGroups.has(group.name);
                const counts = countQueries(group.queries);
                const hasMultiTurn = counts.conversations > 0;

                return (
                  <div key={group.name} className="query-group">
                    <button
                      className="query-group-header"
                      onClick={() => toggleGroup(group.name)}
                      style={{
                        background: meta.bg,
                        borderColor: meta.border,
                        color: meta.color,
                      }}
                    >
                      <span className="query-group-icon">{meta.icon}</span>
                      <span className="query-group-name">
                        {formatTypeName(group.name)}
                      </span>
                      <span className="query-group-count">
                        {hasMultiTurn
                          ? `${counts.conversations} conv · ${counts.total} turns`
                          : counts.total}
                      </span>
                      <ChevronDown
                        size={14}
                        className={`query-group-chevron ${isExpanded ? "expanded" : ""}`}
                      />
                    </button>

                    {isExpanded && (
                      <div className="query-list">
                        {group.queries.map((queryEntry, idx) => {
                          if (Array.isArray(queryEntry)) {
                            // Multi-turn conversation
                            const convKey = `${group.name}-${idx}`;
                            const isConvExpanded = expandedConversations.has(convKey);

                            return (
                              <div key={idx} className="multi-turn-conversation">
                                <button
                                  className="multi-turn-header"
                                  onClick={() => toggleConversation(convKey)}
                                >
                                  <MessageSquare size={14} className="multi-turn-icon" />
                                  <span className="multi-turn-label">
                                    Conversation {idx + 1}
                                  </span>
                                  <span className="multi-turn-turn-count">
                                    {queryEntry.length} turns
                                  </span>
                                  <ChevronDown
                                    size={12}
                                    className={`query-group-chevron ${isConvExpanded ? "expanded" : ""}`}
                                  />
                                </button>

                                {isConvExpanded && (
                                  <div className="multi-turn-turns">
                                    {queryEntry.map((turnQuery, turnIdx) => {
                                      const opMeta = detectOperation(turnQuery);
                                      return (
                                        <div key={turnIdx} className="multi-turn-item">
                                          <div className="multi-turn-item-header">
                                            <span className="multi-turn-number">
                                              T{turnIdx + 1}
                                            </span>
                                            {opMeta && (
                                              <span
                                                className="multi-turn-op-badge"
                                                style={{
                                                  color: opMeta.color,
                                                  borderColor: opMeta.color,
                                                }}
                                              >
                                                {opMeta.label}
                                              </span>
                                            )}
                                          </div>
                                          <span className="query-text">{turnQuery}</span>
                                        </div>
                                      );
                                    })}
                                  </div>
                                )}
                              </div>
                            );
                          }

                          // Single-turn query (string)
                          return (
                            <div key={idx} className="query-item">
                              <span className="query-number">{idx + 1}</span>
                              <span className="query-text">{queryEntry}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
