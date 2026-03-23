# XML Viewer

A Vite + React + TypeScript UI for visualizing XML tree structures used in the Semantic XPath pipeline — itineraries, to-do lists, and other structured data.

## Features

- **Drag & drop** or file-picker to load any `.xml` file
- **Schema-aware views**: specialized layouts for Itinerary and ToDoList schemas
- **Generic fallback**: tree view for any other XML structure
- **Version support**: view different versions of versioned XML trees

## Quick Start

```bash
# From project root
cd tools/xml_viewer

# Install dependencies
npm install

# Start dev server
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) and load an XML file.

## Usage

You can load XML files from:
- `storage/memory/` — source data files (e.g., `todolist.xml`, `travel_toronto_10day.xml`)
- `experiment/experiment_result/.../Session_N/tree.xml` — versioned session trees

## Project Structure

```
src/
├── App.tsx                    # Main app with file loading
├── components/
│   ├── FileInput.tsx          # Drag & drop file input
│   ├── GenericView.tsx        # Generic XML tree viewer
│   ├── ItineraryView.tsx      # Itinerary-specific layout
│   └── ToDoListView.tsx       # ToDoList-specific layout
├── utils/
│   └── xmlParser.ts           # XML parsing utilities
├── types.ts                   # TypeScript type definitions
├── index.css                  # Styles
└── App.css                    # App-specific styles
```

## Tech Stack

- [Vite](https://vitejs.dev/) — build tool
- [React 19](https://react.dev/) — UI framework
- [TypeScript](https://www.typescriptlang.org/) — type safety
- [fast-xml-parser](https://github.com/NaturalIntelligence/fast-xml-parser) — XML parsing
- [Lucide React](https://lucide.dev/) — icons
