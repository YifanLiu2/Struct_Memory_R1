import React, { useState, useEffect, useCallback } from "react";
import {
  FlaskConical,
  ChevronDown,
  Search,
  PlusCircle,
  Pencil,
  Trash2,
  MessageSquare,
  HelpCircle,
  Loader2,
  AlertCircle,
  TreePine,
  ArrowLeft,
} from "lucide-react";

/* ─── Types ─── */

interface QueryInfo {
  folderName: string;
  treePath: string | null;
  logPath: string | null;
  userQuery?: string;
  operationType?: string;
}

interface Session {
  name: string;
  originalTreePath: string | null;
  queries: QueryInfo[];
}

interface ExperimentStructure {
  name: string;
  sessions: Session[];
}

interface ExperimentBrowserProps {
  onXmlLoaded: (content: string, selectLast?: boolean) => void;
  onQueryLogLoaded?: (log: any) => void;
  onCrudTraceLoaded?: (trace: any) => void;
  onExperimentSelected?: (name: string, structure: ExperimentStructure) => void;
  initialExperiment?: string;
  mode?: "welcome" | "sidebar";
}

/* ─── Session styling metadata ─── */

const SESSION_META: Record<
  string,
  {
    icon: React.ReactNode;
    color: string;
    bg: string;
    border: string;
    label: string;
  }
> = {
  create_queries: {
    icon: <PlusCircle size={15} />,
    color: "#166534",
    bg: "#f0fdf4",
    border: "#bbf7d0",
    label: "Create",
  },
  delete_queries: {
    icon: <Trash2 size={15} />,
    color: "#991b1b",
    bg: "#fef2f2",
    border: "#fecaca",
    label: "Delete",
  },
  read_queries: {
    icon: <Search size={15} />,
    color: "#0369a1",
    bg: "#f0f9ff",
    border: "#bae6fd",
    label: "Read",
  },
  update_queries: {
    icon: <Pencil size={15} />,
    color: "#854d0e",
    bg: "#fefce8",
    border: "#fde68a",
    label: "Update",
  },
};

const MULTI_TURN_META = {
  icon: <MessageSquare size={15} />,
  color: "#6d28d9",
  bg: "#f5f3ff",
  border: "#c4b5fd",
  label: "Multi-Turn",
};

const DEFAULT_META = {
  icon: <HelpCircle size={15} />,
  color: "#475569",
  bg: "#f8fafc",
  border: "#e2e8f0",
  label: "Other",
};

function getSessionMeta(name: string) {
  if (SESSION_META[name]) return SESSION_META[name];
  if (name.startsWith("multi-turn") || name.startsWith("multi_turn"))
    return {
      ...MULTI_TURN_META,
      label: name
        .replace(/[-_]/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase()),
    };
  return {
    ...DEFAULT_META,
    label: name.replace(/[-_]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
  };
}

/** Format experiment name: "itinerary_experiment" → "Itinerary" */
function formatExperimentName(name: string): string {
  return name
    .replace(/_experiment.*$/, "")
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ─── Operation badge colors ─── */
const OP_COLORS: Record<string, { color: string; bg: string }> = {
  CREATE: { color: "#166534", bg: "#dcfce7" },
  DELETE: { color: "#991b1b", bg: "#fee2e2" },
  UPDATE: { color: "#854d0e", bg: "#fef9c3" },
  READ: { color: "#0369a1", bg: "#e0f2fe" },
};

/* ─── Component ─── */

export const ExperimentBrowser: React.FC<ExperimentBrowserProps> = ({
  onXmlLoaded,
  onQueryLogLoaded,
  onCrudTraceLoaded,
  onExperimentSelected,
  initialExperiment,
  mode = "welcome",
}) => {
  const [experiments, setExperiments] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(
    initialExperiment || null,
  );
  const [structure, setStructure] = useState<ExperimentStructure | null>(null);
  const [loadingStructure, setLoadingStructure] = useState(false);

  const [expandedSessions, setExpandedSessions] = useState<Set<string>>(
    new Set(),
  );
  const [activeQueryPath, setActiveQueryPath] = useState<string | null>(null);
  const [loadingQuery, setLoadingQuery] = useState<string | null>(null);

  // Fetch experiment list
  useEffect(() => {
    fetch("/api/experiments")
      .then((res) => res.json())
      .then((data: string[]) => {
        setExperiments(data);
        setLoading(false);
      })
      .catch((err) => {
        setError("Failed to load experiments: " + err.message);
        setLoading(false);
      });
  }, []);

  // Auto-load experiment structure when initialExperiment is set
  useEffect(() => {
    if (initialExperiment && !structure) {
      loadExperimentStructure(initialExperiment);
    }
  }, [initialExperiment]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadExperimentStructure = async (name: string) => {
    setSelectedExperiment(name);
    setLoadingStructure(true);
    setError(null);
    setStructure(null);
    setActiveQueryPath(null);

    try {
      const res = await fetch(
        `/api/experiment-structure?name=${encodeURIComponent(name)}`,
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: ExperimentStructure = await res.json();
      setStructure(data);
      setExpandedSessions(new Set(data.sessions.map((s) => s.name)));
      if (onExperimentSelected) {
        onExperimentSelected(name, data);
      }
    } catch (err: any) {
      setError("Failed to load experiment: " + err.message);
    } finally {
      setLoadingStructure(false);
    }
  };

  // Find the crud trace file path for a query
  const findCrudTracePath = (query: QueryInfo): string | null => {
    if (!query.logPath) return null;
    // logPath looks like: "experiment/.../01_Query_.../query_master_log.json"
    // crud trace is at:   "experiment/.../01_Query_.../reasoning_traces/crud_*.json"
    const basePath = query.logPath.replace("/query_master_log.json", "");
    return basePath;
  };

  // Load an XML file (original tree or query result)
  const handleLoadXml = useCallback(
    async (xmlPath: string, logPath?: string | null, query?: QueryInfo) => {
      setLoadingQuery(xmlPath);
      setActiveQueryPath(xmlPath);
      setError(null);

      try {
        const res = await fetch(
          `/api/xml-content?path=${encodeURIComponent(xmlPath)}`,
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const content = await res.text();

        // Load query log if available
        let logData = null;
        if (logPath) {
          try {
            const logRes = await fetch(
              `/api/experiment-log?path=${encodeURIComponent(logPath)}`,
            );
            if (logRes.ok) {
              logData = await logRes.json();
            }
          } catch {
            /* ignore */
          }
        }

        // Load CRUD trace if available
        let traceData = null;
        if (query) {
          const traceBasePath = findCrudTracePath(query);
          if (traceBasePath) {
            try {
              const traceRes = await fetch(
                `/api/experiment-crud-trace?basePath=${encodeURIComponent(traceBasePath)}`,
              );
              if (traceRes.ok) {
                traceData = await traceRes.json();
              }
            } catch {
              /* ignore */
            }
          }
        }

        // selectLast=true for query results (show modified version), false for original trees
        const isQueryResult = !!logPath;
        onXmlLoaded(content, isQueryResult);
        if (onQueryLogLoaded) onQueryLogLoaded(logData);
        if (onCrudTraceLoaded) onCrudTraceLoaded(traceData);
      } catch (err: any) {
        setError("Failed to load file: " + err.message);
      } finally {
        setLoadingQuery(null);
      }
    },
    [onXmlLoaded, onQueryLogLoaded, onCrudTraceLoaded],
  );

  const toggleSession = (name: string) => {
    setExpandedSessions((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const isSidebar = mode === "sidebar";

  // ─── Loading state ───
  if (loading) {
    return (
      <div className={`exp-browser ${isSidebar ? "sidebar-mode" : ""}`}>
        <div className="exp-browser-loading">
          <Loader2 size={24} className="spinning" />
          <span>Scanning experiments…</span>
        </div>
      </div>
    );
  }

  // ─── Error (fatal) ───
  if (error && !selectedExperiment) {
    return (
      <div className={`exp-browser ${isSidebar ? "sidebar-mode" : ""}`}>
        <div className="exp-browser-error">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  // ─── Experiment card grid (welcome mode only) ───
  if (!selectedExperiment && !isSidebar) {
    return (
      <div className="exp-browser">
        <div className="exp-browser-header">
          <FlaskConical size={18} />
          <span>Select Experiment</span>
        </div>
        <div className="exp-browser-grid">
          {experiments.map((name) => (
            <button
              key={name}
              className="exp-card"
              onClick={() => loadExperimentStructure(name)}
            >
              <FlaskConical size={22} className="exp-card-icon" />
              <span className="exp-card-name">
                {formatExperimentName(name)}
              </span>
              <span className="exp-card-raw">{name}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  // ─── Session/query list (both welcome & sidebar modes) ───
  return (
    <div className={`exp-browser ${isSidebar ? "sidebar-mode" : ""}`}>
      <div className="exp-browser-header">
        {!isSidebar && (
          <button
            className="exp-back-btn"
            onClick={() => {
              setSelectedExperiment(null);
              setStructure(null);
              setActiveQueryPath(null);
              if (onQueryLogLoaded) onQueryLogLoaded(null);
              if (onCrudTraceLoaded) onCrudTraceLoaded(null);
            }}
          >
            <ArrowLeft size={16} />
          </button>
        )}
        <FlaskConical size={18} />
        <span>{formatExperimentName(selectedExperiment!)}</span>
      </div>

      {loadingStructure ? (
        <div className="exp-browser-loading">
          <Loader2 size={20} className="spinning" />
          <span>Loading…</span>
        </div>
      ) : structure ? (
        <div className="exp-session-list">
          {error && (
            <div className="exp-browser-error compact">
              <AlertCircle size={14} />
              <span>{error}</span>
            </div>
          )}

          {structure.sessions.map((session) => {
            const meta = getSessionMeta(session.name);
            const isExpanded = expandedSessions.has(session.name);

            return (
              <div key={session.name} className="exp-session">
                <button
                  className="exp-session-header"
                  onClick={() => toggleSession(session.name)}
                  style={{
                    borderLeftColor: meta.border,
                    background: meta.bg,
                  }}
                >
                  <span
                    className="exp-session-icon"
                    style={{ color: meta.color }}
                  >
                    {meta.icon}
                  </span>
                  <span
                    className="exp-session-name"
                    style={{ color: meta.color }}
                  >
                    {meta.label}
                  </span>
                  <span className="exp-session-count">
                    {session.queries.length}
                  </span>
                  <ChevronDown
                    size={14}
                    className={`exp-session-chevron ${isExpanded ? "expanded" : ""}`}
                    style={{ color: meta.color }}
                  />
                </button>

                {isExpanded && (
                  <div className="exp-session-content">
                    {/* Original tree */}
                    {session.originalTreePath && (
                      <button
                        className={`exp-query-item exp-original-tree ${
                          activeQueryPath === session.originalTreePath
                            ? "active"
                            : ""
                        }`}
                        onClick={() => {
                          handleLoadXml(session.originalTreePath!);
                        }}
                        disabled={loadingQuery === session.originalTreePath}
                      >
                        {loadingQuery === session.originalTreePath ? (
                          <Loader2 size={14} className="spinning" />
                        ) : (
                          <TreePine size={14} className="exp-original-icon" />
                        )}
                        <span className="exp-query-text">Original Tree</span>
                      </button>
                    )}

                    {/* Query items */}
                    {session.queries.map((query, idx) => {
                      const isActive = activeQueryPath === query.treePath;
                      const isLoading = loadingQuery === query.treePath;
                      const displayText =
                        query.userQuery ||
                        query.folderName
                          .replace(/^\d+_Query_/, "")
                          .replace(/_/g, " ");
                      const opColor = query.operationType
                        ? OP_COLORS[query.operationType]
                        : null;

                      return (
                        <button
                          key={query.folderName}
                          className={`exp-query-item ${isActive ? "active" : ""}`}
                          onClick={() => {
                            if (query.treePath) {
                              handleLoadXml(
                                query.treePath,
                                query.logPath,
                                query,
                              );
                            }
                          }}
                          disabled={!query.treePath || isLoading}
                          title={query.userQuery || displayText}
                        >
                          {isLoading ? (
                            <Loader2 size={14} className="spinning" />
                          ) : (
                            <span className="exp-query-num">{idx + 1}</span>
                          )}
                          <span className="exp-query-text">{displayText}</span>
                          {opColor && query.operationType && (
                            <span
                              className="exp-op-badge"
                              style={{
                                color: opColor.color,
                                background: opColor.bg,
                              }}
                            >
                              {query.operationType}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
};
