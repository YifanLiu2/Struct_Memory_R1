import { useState, useCallback, useEffect } from "react";
import { parseXML } from "./utils/xmlParser";
import type { ParsedData, ParsedDocument } from "./utils/xmlParser";
import { FileInput } from "./components/FileInput";
import { FileBrowser } from "./components/FileBrowser";
import { ExperimentBrowser } from "./components/ExperimentBrowser";
import { ItineraryView } from "./components/ItineraryView";
import { ToDoListView } from "./components/ToDoListView";
import { MealPlanView } from "./components/MealPlanView";
import { GenericView } from "./components/GenericView";
import { QueryPanel } from "./components/QueryPanel";
import {
  RotateCcw,
  Upload,
  FolderOpen,
  ArrowLeft,
  MessageSquare,
  Search,
  Crosshair,
  Clock,
  CheckCircle2,
  XCircle,
  Trash2,
  PlusCircle,
  Pencil,
  ChevronLeft,
  ChevronRight,
  History,
} from "lucide-react";
import "./App.css";

/* ─── Types for experiment sidebar state ─── */
interface ExperimentContext {
  experimentName: string;
  structure: any; // ExperimentStructure from API
}

/* ─── Operation badge helpers ─── */
const OP_ICON: Record<string, React.ReactNode> = {
  CREATE: <PlusCircle size={14} />,
  DELETE: <Trash2 size={14} />,
  UPDATE: <Pencil size={14} />,
  READ: <Search size={14} />,
};
const OP_COLOR: Record<string, string> = {
  CREATE: "#166534",
  DELETE: "#991b1b",
  UPDATE: "#854d0e",
  READ: "#0369a1",
};

function App() {
  const [documents, setDocuments] = useState<ParsedData>([]);
  const [selectedDocIndex, setSelectedDocIndex] = useState<number>(0);
  const [showManualInput, setShowManualInput] = useState(false);
  const [showFileBrowser, setShowFileBrowser] = useState(false);

  // Experiment context — persists while viewing query results
  const [experimentCtx, setExperimentCtx] = useState<ExperimentContext | null>(
    null,
  );
  const [queryLog, setQueryLog] = useState<any>(null);
  const [crudTrace, setCrudTrace] = useState<any>(null);

  const handleXmlLoaded = useCallback(
    (content: string, selectLast?: boolean) => {
      const result = parseXML(content);
      setDocuments(result);
      setSelectedDocIndex(
        selectLast && result.length > 0 ? result.length - 1 : 0,
      );
      setShowManualInput(false);
      setShowFileBrowser(false);
    },
    [],
  );

  const handleQueryLogLoaded = useCallback((log: any) => {
    setQueryLog(log);
  }, []);

  const handleCrudTraceLoaded = useCallback((trace: any) => {
    setCrudTrace(trace);
  }, []);

  // Auto-select version based on query log
  useEffect(() => {
    if (!queryLog || documents.length <= 1) return;

    // Look for tree_version (result) or resolved_version (target)
    // If we have a resulting tree version, that's usually what we want to see.
    // If not, we might want to see the version being operated on.
    const targetVersion =
      queryLog.treeVersion?.version ||
      queryLog.version_result?.resolved_version;

    if (targetVersion) {
      // Find document with matching version number
      // parsed doc data has "@_number" from xmlParser, or fallback to index+1
      const index = documents.findIndex((doc, i) => {
        const docVersion = doc.data["@_number"] || i + 1;
        return docVersion == targetVersion;
      });

      if (index !== -1) {
        setSelectedDocIndex(index);
      }
    }
  }, [queryLog, documents]);

  const handleExperimentSelected = useCallback(
    (name: string, structure: any) => {
      setExperimentCtx({ experimentName: name, structure });
    },
    [],
  );

  const handleReset = () => {
    setDocuments([]);
    setSelectedDocIndex(0);
    setShowManualInput(false);
    setShowFileBrowser(false);
    setExperimentCtx(null);
    setQueryLog(null);
    setCrudTrace(null);
  };

  const handleBack = () => {
    setDocuments([]);
    setSelectedDocIndex(0);
    setQueryLog(null);
    setCrudTrace(null);
    // keep experimentCtx so the sidebar stays
  };

  const currentDoc: ParsedDocument | undefined = documents[selectedDocIndex];
  const isViewingResults = documents.length > 0;

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo">XML Viewer</div>

        {/* Version Switcher */}
        {documents.length > 1 && (
          <div className="version-switcher">
            <button
              className="version-btn"
              disabled={selectedDocIndex === 0}
              onClick={() => setSelectedDocIndex((i) => i - 1)}
              title="Previous Version"
            >
              <ChevronLeft size={16} />
            </button>

            <div className="version-info">
              <History size={14} className="version-icon" />
              <span className="version-label">
                {currentDoc?.label || `Version ${selectedDocIndex + 1}`}
              </span>
              <span className="version-count">of {documents.length}</span>
            </div>

            <button
              className="version-btn"
              disabled={selectedDocIndex === documents.length - 1}
              onClick={() => setSelectedDocIndex((i) => i + 1)}
              title="Next Version"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        )}

        <div className="header-actions">
          {experimentCtx && (
            <button
              className="reset-btn"
              onClick={isViewingResults ? handleBack : handleReset}
            >
              <ArrowLeft size={16} /> Back
            </button>
          )}
          {(isViewingResults || experimentCtx) && (
            <button className="reset-btn" onClick={handleReset}>
              <RotateCcw size={16} /> Load New
            </button>
          )}
        </div>
      </header>

      {/* Layout: sidebar + main content when experiment is active */}
      <div className={`app-body ${experimentCtx ? "with-sidebar" : ""}`}>
        {/* Experiment sidebar — visible when an experiment is selected */}
        {experimentCtx && (
          <aside className="experiment-sidebar">
            <ExperimentBrowser
              onXmlLoaded={handleXmlLoaded}
              onQueryLogLoaded={handleQueryLogLoaded}
              onCrudTraceLoaded={handleCrudTraceLoaded}
              onExperimentSelected={handleExperimentSelected}
              initialExperiment={experimentCtx.experimentName}
              mode="sidebar"
            />
          </aside>
        )}

        <main
          className={`app-content ${experimentCtx ? "sidebar-active" : ""}`}
        >
          {!isViewingResults ? (
            /* ─── Welcome screen ─── */
            !experimentCtx ? (
              <div className="welcome-screen">
                <h1>XML Viewer</h1>
                <p className="subtitle">
                  Browse experiment results, or upload/paste XML manually
                </p>
                <div className="input-layout">
                  <div className="input-layout-primary">
                    <ExperimentBrowser
                      onXmlLoaded={handleXmlLoaded}
                      onQueryLogLoaded={handleQueryLogLoaded}
                      onCrudTraceLoaded={handleCrudTraceLoaded}
                      onExperimentSelected={handleExperimentSelected}
                      mode="welcome"
                    />
                  </div>
                  <div className="input-layout-secondary">
                    {!showFileBrowser && !showManualInput && (
                      <div className="secondary-options">
                        <button
                          className="show-manual-btn"
                          onClick={() => setShowFileBrowser(true)}
                        >
                          <FolderOpen size={16} />
                          Browse All XML Files
                        </button>
                        <button
                          className="show-manual-btn"
                          onClick={() => setShowManualInput(true)}
                        >
                          <Upload size={16} />
                          Upload or Paste XML
                        </button>
                      </div>
                    )}
                    {showFileBrowser && (
                      <>
                        <button
                          className="hide-manual-btn"
                          onClick={() => setShowFileBrowser(false)}
                        >
                          ← Back to experiments
                        </button>
                        <FileBrowser onXmlLoaded={handleXmlLoaded} />
                      </>
                    )}
                    {showManualInput && (
                      <>
                        <button
                          className="hide-manual-btn"
                          onClick={() => setShowManualInput(false)}
                        >
                          ← Back to experiments
                        </button>
                        <FileInput onXmlLoaded={handleXmlLoaded} />
                      </>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              /* Experiment selected but no query chosen yet — show prompt */
              <div className="welcome-screen">
                <h1>XML Viewer</h1>
                <p className="subtitle">
                  Select a query from the sidebar to view the result
                </p>
              </div>
            )
          ) : (
            /* ─── Viewing results ─── */
            <div className="view-wrapper">
              {/* Query metadata banner */}
              {queryLog && (
                <div className="query-meta-banner">
                  <div className="query-meta-row query-meta-query">
                    <MessageSquare size={15} className="query-meta-icon" />
                    <div className="query-meta-content">
                      <span className="query-meta-label">Query</span>
                      <span className="query-meta-value italic">
                        "{queryLog.user_query}"
                      </span>
                    </div>
                  </div>

                  <div className="query-meta-row-group">
                    {queryLog.operation_type && (
                      <div className="query-meta-chip">
                        <span
                          className="query-meta-op-icon"
                          style={{
                            color:
                              OP_COLOR[queryLog.operation_type] || "#475569",
                          }}
                        >
                          {OP_ICON[queryLog.operation_type] || null}
                        </span>
                        <span
                          className="query-meta-op-label"
                          style={{
                            color:
                              OP_COLOR[queryLog.operation_type] || "#475569",
                          }}
                        >
                          {queryLog.operation_type}
                        </span>
                      </div>
                    )}
                    {queryLog.semantic_xpath_query && (
                      <div className="query-meta-chip xpath-chip">
                        <Crosshair size={13} />
                        <code>{queryLog.semantic_xpath_query}</code>
                      </div>
                    )}
                  </div>

                  {/* Reasoning from CRUD trace */}
                  {crudTrace && (
                    <div className="query-meta-reasoning-section">
                      {renderCrudReasoning(crudTrace, queryLog)}
                    </div>
                  )}

                  {/* Task result summary */}
                  {queryLog.downstream_task_result && (
                    <div className="query-meta-row query-meta-result">
                      {queryLog.downstream_task_result.success ? (
                        <CheckCircle2
                          size={15}
                          className="query-meta-icon success"
                        />
                      ) : (
                        <XCircle size={15} className="query-meta-icon error" />
                      )}
                      <div className="query-meta-content">
                        <span className="query-meta-label">Result</span>
                        <span className="query-meta-value">
                          {queryLog.downstream_task_result.success
                            ? formatResultSummary(queryLog)
                            : "Operation failed"}
                        </span>
                      </div>
                      {queryLog.execution_time_ms && (
                        <span className="query-meta-time">
                          <Clock size={12} />
                          {(queryLog.execution_time_ms / 1000).toFixed(1)}s
                        </span>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="view-container">
                {currentDoc?.type === "itinerary" && (
                  <ItineraryView data={currentDoc.data} />
                )}
                {currentDoc?.type === "todolist" && (
                  <ToDoListView data={currentDoc.data} />
                )}
                {currentDoc?.type === "mealplan" && (
                  <MealPlanView data={currentDoc.data} />
                )}
                {currentDoc?.type === "generic" && (
                  <GenericView data={currentDoc.data} />
                )}
              </div>
            </div>
          )}
        </main>
      </div>
      <QueryPanel />
    </div>
  );
}

/* ─── Render CRUD reasoning from trace log ─── */
function renderCrudReasoning(trace: any, _log: any): React.ReactNode {
  const op = trace.operation;

  if (op === "CREATE" && trace.operation_data?.insertion_point?.reasoning) {
    return (
      <div className="query-meta-row query-meta-reasoning">
        <PlusCircle
          size={14}
          className="query-meta-icon"
          style={{ color: "#166534" }}
        />
        <div className="query-meta-content">
          <span className="query-meta-label">Reasoning</span>
          <span className="query-meta-value">
            {trace.operation_data.insertion_point.reasoning}
          </span>
          {trace.operation_data.created_path && (
            <span className="query-meta-path-detail">
              Created at: <code>{trace.operation_data.created_path}</code>
            </span>
          )}
        </div>
      </div>
    );
  }

  if (op === "DELETE" && trace.operation_data?.deleted_paths?.length > 0) {
    return (
      <div className="query-meta-row query-meta-reasoning">
        <Trash2
          size={14}
          className="query-meta-icon"
          style={{ color: "#991b1b" }}
        />
        <div className="query-meta-content">
          <span className="query-meta-label">Deleted Nodes</span>
          <div className="query-meta-paths-list">
            {trace.operation_data.deleted_paths.map((p: string, i: number) => (
              <span key={i} className="query-meta-path-item">
                <span className="query-meta-path-bullet">•</span>
                <code>{p}</code>
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (op === "UPDATE" && trace.operation_data?.update_results?.length > 0) {
    return (
      <div className="query-meta-row query-meta-reasoning">
        <Pencil
          size={14}
          className="query-meta-icon"
          style={{ color: "#854d0e" }}
        />
        <div className="query-meta-content">
          <span className="query-meta-label">
            Changes ({trace.operation_data.updated_count} nodes)
          </span>
          <div className="query-meta-changes-list">
            {trace.operation_data.update_results
              .slice(0, 3)
              .map((result: any, i: number) => (
                <div key={i} className="query-meta-change-item">
                  <code className="query-meta-change-path">{result.path}</code>
                  {result.changes &&
                    Object.entries(result.changes)
                      .slice(0, 2)
                      .map(([field, change]: [string, any]) => (
                        <div key={field} className="query-meta-change-field">
                          <span className="query-meta-change-field-name">
                            {field}:
                          </span>
                          <span className="query-meta-change-from">
                            {typeof change.from === "string"
                              ? change.from
                              : JSON.stringify(change.from)}
                          </span>
                          <span className="query-meta-change-arrow">→</span>
                          <span className="query-meta-change-to">
                            {typeof change.to === "string"
                              ? change.to
                              : JSON.stringify(change.to)}
                          </span>
                        </div>
                      ))}
                </div>
              ))}
            {trace.operation_data.update_results.length > 3 && (
              <span className="query-meta-more">
                +{trace.operation_data.update_results.length - 3} more…
              </span>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (op === "READ" && trace.operation_data?.selected_nodes?.length > 0) {
    return (
      <div className="query-meta-row query-meta-reasoning">
        <Search
          size={14}
          className="query-meta-icon"
          style={{ color: "#0369a1" }}
        />
        <div className="query-meta-content">
          <span className="query-meta-label">
            Selected Nodes ({trace.operation_data.selected_count})
          </span>
          <div className="query-meta-paths-list">
            {trace.operation_data.selected_nodes.map((node: any, i: number) => (
              <div key={i} className="query-meta-read-node">
                <span className="query-meta-path-item">
                  <span className="query-meta-path-bullet">•</span>
                  <code>{node.tree_path}</code>
                </span>
                {node.reasoning && (
                  <span className="query-meta-node-reasoning">
                    {node.reasoning}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return null;
}

/** Build a human-readable result summary from the query log */
function formatResultSummary(log: any): string {
  const result = log.downstream_task_result;
  const op = log.operation_type;

  if (op === "DELETE" && result.deleted_count != null) {
    return `Deleted ${result.deleted_count} node${result.deleted_count !== 1 ? "s" : ""}`;
  }
  if (op === "CREATE") {
    return `Created node at ${result.created_path || "target location"}`;
  }
  if (op === "UPDATE" && result.updated_count != null) {
    return `Updated ${result.updated_count} node${result.updated_count !== 1 ? "s" : ""}`;
  }
  if (op === "READ" && result.selected_count != null) {
    return `Found ${result.selected_count} match${result.selected_count !== 1 ? "es" : ""}`;
  }
  return "Completed successfully";
}

export default App;
