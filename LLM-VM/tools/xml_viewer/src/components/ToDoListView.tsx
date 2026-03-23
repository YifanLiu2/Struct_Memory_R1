import React from "react";
import {
  CheckCircle2,
  Circle,
  AlertCircle,
  Calendar,
  StickyNote,
  MessageSquare,
  GitBranch,
} from "lucide-react";
import type { ToDoListData, Project, Task } from "../types";
import { ensureArray } from "../utils/xmlParser";

interface ToDoListViewProps {
  data: ToDoListData;
}

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const PriorityBadge: React.FC<{ priority: string }> = ({ priority }) => {
  const normalized = capitalize(priority.toLowerCase());
  const colors: Record<string, string> = {
    High: "bg-red-100 text-red-800",
    Medium: "bg-yellow-100 text-yellow-800",
    Low: "bg-green-100 text-green-800",
  };
  const colorClass = colors[normalized] || "bg-gray-100 text-gray-800";

  return <span className={`priority-badge ${colorClass}`}>{normalized}</span>;
};

const TaskItem: React.FC<{ task: Task }> = ({ task }) => {
  const status = task.status || "Todo";
  const description =
    task.description || (task as any).title || "Untitled task";
  const priority = task.priority || "Medium";
  const dueDate = task.due_date || "";
  const isDone =
    status.toLowerCase() === "done" || status.toLowerCase() === "completed";
  const isBlocked = status.toLowerCase() === "blocked";

  return (
    <div className={`task-item ${isBlocked ? "blocked" : ""}`}>
      <div className="task-status-icon">
        {isDone ? (
          <CheckCircle2 size={20} className="text-green-500" />
        ) : (
          <Circle size={20} className="text-gray-400" />
        )}
      </div>
      <div className="task-content">
        <div className="task-header">
          <span
            className={`task-text ${isDone ? "line-through text-gray-500" : ""}`}
          >
            {description}
          </span>
          <PriorityBadge priority={priority} />
        </div>
        <div className="task-meta">
          {dueDate && (
            <div className="meta-item">
              <Calendar size={14} />
              <span>Due: {dueDate}</span>
            </div>
          )}
          <span
            className={`task-status-text ${isBlocked ? "text-red-600" : ""}`}
          >
            {capitalize(status)}
          </span>
        </div>
        {task.note && (
          <div className="task-note">
            <StickyNote size={14} className="inline mr-1" />
            {task.note}
          </div>
        )}
      </div>
    </div>
  );
};

const ProjectCard: React.FC<{ project: Project }> = ({ project }) => {
  const tasks = ensureArray(project.Task);
  const status = project.status || "Active";
  const statusSlug = status.toLowerCase().replace(" ", "-");

  return (
    <div className="project-card">
      <div className="project-header">
        <h3 className="project-title">{project.name}</h3>
        <div className="project-meta-top">
          <span className={`status-pill ${statusSlug}`}>{status}</span>
          {project.deadline && (
            <div className="meta-item warning">
              <AlertCircle size={14} />
              <span>Deadline: {project.deadline}</span>
            </div>
          )}
        </div>
      </div>

      <div className="project-tasks">
        {tasks.map((task, idx) => (
          <TaskItem key={idx} task={task} />
        ))}
      </div>
    </div>
  );
};

const VersionMeta: React.FC<{ data: ToDoListData }> = ({ data }) => {
  const conversationHistory = data.conversation_history;
  const patchInfo = data.patch_info;

  // Don't render if both are empty/missing
  if (!conversationHistory && !patchInfo) return null;

  // Check if values are truly empty (XML self-closing tags parse as empty strings or objects)
  const hasConversation =
    conversationHistory &&
    typeof conversationHistory === "string" &&
    conversationHistory.trim().length > 0;
  const hasPatch =
    patchInfo && typeof patchInfo === "string" && patchInfo.trim().length > 0;

  if (!hasConversation && !hasPatch) return null;

  /**
   * Parse patch_info into a prefix and individual paths.
   * Format examples:
   *   "Deleted: Root > ... > Task 2"
   *   "Deleted 5 nodes: Root > ... > Task 2, Root > ... > Task 1, ..."
   *   "Updated: Root > ... > Task 1"
   *   "Created: Root > ... > Task 3"
   */
  const parsePatchInfo = (text: string) => {
    // Match prefix like "Deleted 5 nodes:" or "Deleted:" or "Updated:" etc.
    const prefixMatch = text.match(
      /^((?:Deleted|Updated|Created|Modified)(?:\s+\d+\s+nodes?)?)\s*:\s*/i,
    );
    if (!prefixMatch) return { prefix: null, paths: [text] };

    const prefix = prefixMatch[1];
    const remainder = text.slice(prefixMatch[0].length);

    // Split on ", Root" to separate paths (keeping "Root" on each)
    const paths = remainder
      .split(/,\s*(?=Root\s*>)/i)
      .map((p) => p.trim())
      .filter((p) => p.length > 0);

    return { prefix, paths };
  };

  return (
    <div className="version-meta-banner">
      {hasConversation && (
        <div className="version-meta-item conversation">
          <div className="version-meta-icon">
            <MessageSquare size={16} />
          </div>
          <div className="version-meta-content">
            <span className="version-meta-label">Query</span>
            <span className="version-meta-value">"{conversationHistory}"</span>
          </div>
        </div>
      )}
      {hasPatch &&
        (() => {
          const { prefix, paths } = parsePatchInfo(patchInfo!);
          return (
            <div className="version-meta-item patch">
              <div className="version-meta-icon">
                <GitBranch size={16} />
              </div>
              <div className="version-meta-content">
                <span className="version-meta-label">Changes</span>
                {prefix && (
                  <span className="version-meta-patch-prefix">{prefix}</span>
                )}
                <div className="version-meta-paths">
                  {paths.map((path, idx) => (
                    <div key={idx} className="version-meta-path-line">
                      <span className="version-meta-path-bullet">›</span>
                      <span className="version-meta-path-text">{path}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          );
        })()}
    </div>
  );
};

export const ToDoListView: React.FC<ToDoListViewProps> = ({ data }) => {
  const categories = ensureArray(data.Category);

  return (
    <div className="todolist-view">
      <h1 className="view-title">
        ToDo List {data["@_number"] ? `(v${data["@_number"]})` : ""}
      </h1>
      <VersionMeta data={data} />
      {categories.map((category) => (
        <div key={category["@_name"]} className="category-section">
          <h2 className="category-title">{category["@_name"]}</h2>
          <p className="category-description">{category.description}</p>
          <div className="projects-grid">
            {ensureArray(category.Project).map((project, idx) => (
              <ProjectCard key={idx} project={project} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};
