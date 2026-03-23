import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Resolve the project root (two levels up from tools/xml_viewer)
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..", "..");

interface FileNode {
  name: string;
  relativePath: string;
  type: "file" | "directory";
  children?: FileNode[];
}

function scanDirectory(dir: string, basePath: string): FileNode[] {
  const results: FileNode[] = [];
  if (!fs.existsSync(dir)) return results;

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    const relPath = path.relative(basePath, fullPath).replace(/\\/g, "/");

    if (entry.isDirectory()) {
      const children = scanDirectory(fullPath, basePath);
      // Only include directories that contain XML files (directly or nested)
      if (children.length > 0) {
        results.push({
          name: entry.name,
          relativePath: relPath,
          type: "directory",
          children,
        });
      }
    } else if (entry.isFile() && entry.name.endsWith(".xml")) {
      results.push({
        name: entry.name,
        relativePath: relPath,
        type: "file",
      });
    }
  }
  return results;
}

function experimentBrowserPlugin() {
  const experimentsDir = path.join(
    projectRoot,
    "experiment",
    "experiment_result",
    "semantic_xpath",
  );

  return {
    name: "experiment-browser-api",
    configureServer(server: any) {
      // List all experiment directories
      server.middlewares.use("/api/experiments", (_req: any, res: any) => {
        const experiments: string[] = [];
        if (fs.existsSync(experimentsDir)) {
          const entries = fs.readdirSync(experimentsDir, {
            withFileTypes: true,
          });
          for (const entry of entries) {
            if (entry.isDirectory()) {
              experiments.push(entry.name);
            }
          }
        }
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(experiments));
      });

      // Get full structure of an experiment (sessions → queries)
      server.middlewares.use(
        "/api/experiment-structure",
        (req: any, res: any) => {
          const queryString = (req.url as string).split("?")[1] || "";
          const params = new URLSearchParams(queryString);
          const name = params.get("name");

          if (!name) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: "Missing name parameter" }));
            return;
          }

          const expDir = path.join(experimentsDir, name);
          if (!expDir.startsWith(experimentsDir) || !fs.existsSync(expDir)) {
            res.statusCode = 404;
            res.end(JSON.stringify({ error: "Experiment not found" }));
            return;
          }

          const sessions: any[] = [];
          const entries = fs.readdirSync(expDir, { withFileTypes: true });

          for (const entry of entries) {
            if (!entry.isDirectory()) continue;
            const sessionDir = path.join(expDir, entry.name);
            const sessionRelPath = path
              .relative(projectRoot, sessionDir)
              .replace(/\\/g, "/");

            // Check for tree.xml (original tree for this session)
            const originalTreePath = path.join(sessionDir, "tree.xml");
            const hasOriginalTree = fs.existsSync(originalTreePath);

            // Scan for query subdirectories
            const queries: any[] = [];
            const sessionEntries = fs.readdirSync(sessionDir, {
              withFileTypes: true,
            });
            for (const qEntry of sessionEntries) {
              if (!qEntry.isDirectory()) continue;
              const qDir = path.join(sessionDir, qEntry.name);
              const qRelPath = path
                .relative(projectRoot, qDir)
                .replace(/\\/g, "/");

              const qTreePath = path.join(qDir, "tree.xml");
              const qLogPath = path.join(qDir, "query_master_log.json");

              const queryInfo: any = {
                folderName: qEntry.name,
                treePath: fs.existsSync(qTreePath)
                  ? qRelPath + "/tree.xml"
                  : null,
                logPath: fs.existsSync(qLogPath)
                  ? qRelPath + "/query_master_log.json"
                  : null,
              };

              // Try to extract basic info from the log for the list view
              if (queryInfo.logPath && fs.existsSync(qLogPath)) {
                try {
                  const logContent = JSON.parse(
                    fs.readFileSync(qLogPath, "utf-8"),
                  );
                  queryInfo.userQuery = logContent.user_query || null;
                  queryInfo.operationType = logContent.operation_type || null;
                } catch {
                  // ignore parse errors
                }
              }

              // Fallback: if query specific tree doesnt exist, use the session tree
              // This is common for versioned experiments
              if (!queryInfo.treePath && hasOriginalTree) {
                queryInfo.treePath = sessionRelPath + "/tree.xml";
                queryInfo.isSharedTree = true;
              }

              queries.push(queryInfo);
            }

            // Sort queries by folder name (they start with ##_ prefix)
            queries.sort((a: any, b: any) =>
              a.folderName.localeCompare(b.folderName),
            );

            sessions.push({
              name: entry.name,
              originalTreePath: hasOriginalTree
                ? sessionRelPath + "/tree.xml"
                : null,
              queries,
            });
          }

          // Sort sessions alphabetically
          sessions.sort((a: any, b: any) => a.name.localeCompare(b.name));

          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ name, sessions }));
        },
      );

      // Read a query_master_log.json file
      server.middlewares.use("/api/experiment-log", (req: any, res: any) => {
        const queryString = (req.url as string).split("?")[1] || "";
        const params = new URLSearchParams(queryString);
        const filePath = params.get("path");

        if (!filePath) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: "Missing path parameter" }));
          return;
        }

        const fullPath = path.resolve(projectRoot, filePath);

        if (
          !fullPath.startsWith(experimentsDir) ||
          !fullPath.endsWith(".json")
        ) {
          res.statusCode = 403;
          res.end(JSON.stringify({ error: "Access denied" }));
          return;
        }

        if (!fs.existsSync(fullPath)) {
          res.statusCode = 404;
          res.end(JSON.stringify({ error: "File not found" }));
          return;
        }

        const content = fs.readFileSync(fullPath, "utf-8");
        res.setHeader("Content-Type", "application/json");
        res.end(content);
      });

      // Read the CRUD reasoning trace for a query
      server.middlewares.use(
        "/api/experiment-crud-trace",
        (req: any, res: any) => {
          const queryString = (req.url as string).split("?")[1] || "";
          const params = new URLSearchParams(queryString);
          const basePath = params.get("basePath");

          if (!basePath) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: "Missing basePath parameter" }));
            return;
          }

          const tracesDir = path.resolve(
            projectRoot,
            basePath,
            "reasoning_traces",
          );

          if (
            !tracesDir.startsWith(experimentsDir) ||
            !fs.existsSync(tracesDir)
          ) {
            res.statusCode = 404;
            res.end(JSON.stringify({ error: "Traces directory not found" }));
            return;
          }

          // Find the first crud_*.json file
          const entries = fs.readdirSync(tracesDir);
          const crudFile = entries.find(
            (e: string) => e.startsWith("crud_") && e.endsWith(".json"),
          );

          if (!crudFile) {
            res.statusCode = 404;
            res.end(JSON.stringify({ error: "No CRUD trace file found" }));
            return;
          }

          const fullPath = path.join(tracesDir, crudFile);
          const content = fs.readFileSync(fullPath, "utf-8");
          res.setHeader("Content-Type", "application/json");
          res.end(content);
        },
      );
    },
  };
}

function queryFilesPlugin() {
  return {
    name: "query-files-api",
    configureServer(server: any) {
      // List YAML query files
      server.middlewares.use("/api/query-files", (_req: any, res: any) => {
        const queriesDir = path.join(projectRoot, "experiment", "queries");
        const files: { name: string; relativePath: string }[] = [];

        if (fs.existsSync(queriesDir)) {
          const entries = fs.readdirSync(queriesDir, { withFileTypes: true });
          for (const entry of entries) {
            if (
              entry.isFile() &&
              (entry.name.endsWith(".yaml") || entry.name.endsWith(".yml"))
            ) {
              files.push({
                name: entry.name,
                relativePath: path
                  .relative(projectRoot, path.join(queriesDir, entry.name))
                  .replace(/\\/g, "/"),
              });
            }
          }
        }

        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(files));
      });

      // Read a specific query file
      server.middlewares.use("/api/query-content", (req: any, res: any) => {
        const queryString = (req.url as string).split("?")[1] || "";
        const params = new URLSearchParams(queryString);
        const filePath = params.get("path");

        if (!filePath) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: "Missing path parameter" }));
          return;
        }

        const fullPath = path.resolve(projectRoot, filePath);
        const allowedDir = path.join(projectRoot, "experiment", "queries");

        if (!fullPath.startsWith(allowedDir)) {
          res.statusCode = 403;
          res.end(JSON.stringify({ error: "Access denied" }));
          return;
        }

        if (!fs.existsSync(fullPath)) {
          res.statusCode = 404;
          res.end(JSON.stringify({ error: "File not found" }));
          return;
        }

        const content = fs.readFileSync(fullPath, "utf-8");
        res.setHeader("Content-Type", "text/plain");
        res.end(content);
      });
    },
  };
}

function xmlFilesPlugin() {
  return {
    name: "xml-files-api",
    configureServer(server: any) {
      server.middlewares.use("/api/xml-files", (_req: any, res: any) => {
        const memoryDir = path.join(projectRoot, "storage", "memory");
        const experimentDir = path.join(
          projectRoot,
          "experiment",
          "experiment_result",
        );

        const tree = [
          {
            name: "Storage / Memory",
            relativePath: "storage/memory",
            type: "directory" as const,
            children: scanDirectory(memoryDir, projectRoot),
          },
          {
            name: "Experiments",
            relativePath: "experiment/experiment_result",
            type: "directory" as const,
            children: scanDirectory(experimentDir, projectRoot),
          },
        ];

        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(tree));
      });

      server.middlewares.use("/api/xml-content", (req: any, res: any) => {
        const queryString = (req.url as string).split("?")[1] || "";
        const params = new URLSearchParams(queryString);
        const filePath = params.get("path");

        if (!filePath) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: "Missing path parameter" }));
          return;
        }

        // Security: only allow paths within allowed directories
        const fullPath = path.resolve(projectRoot, filePath);
        const allowedDirs = [
          path.join(projectRoot, "storage", "memory"),
          path.join(projectRoot, "experiment"),
        ];

        const isAllowed = allowedDirs.some((dir) => fullPath.startsWith(dir));
        if (!isAllowed || !fullPath.endsWith(".xml")) {
          res.statusCode = 403;
          res.end(JSON.stringify({ error: "Access denied" }));
          return;
        }

        if (!fs.existsSync(fullPath)) {
          res.statusCode = 404;
          res.end(JSON.stringify({ error: "File not found" }));
          return;
        }

        const content = fs.readFileSync(fullPath, "utf-8");
        res.setHeader("Content-Type", "text/plain");
        res.end(content);
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    xmlFilesPlugin(),
    queryFilesPlugin(),
    experimentBrowserPlugin(),
  ],
});
