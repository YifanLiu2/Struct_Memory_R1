import React, { useState, useCallback } from "react";
import { Upload, Code } from "lucide-react";

interface FileInputProps {
  onXmlLoaded: (content: string) => void;
}

export const FileInput: React.FC<FileInputProps> = ({ onXmlLoaded }) => {
  const [dragActive, setDragActive] = useState(false);
  const [textInput, setTextInput] = useState("");

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
          if (e.target?.result) {
            onXmlLoaded(e.target.result as string);
          }
        };
        reader.readAsText(file);
      }
    },
    [onXmlLoaded],
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target?.result) {
          onXmlLoaded(e.target.result as string);
        }
      };
      reader.readAsText(file);
    }
  };

  const handleTextSubmit = () => {
    if (textInput.trim()) {
      onXmlLoaded(textInput);
    }
  };

  const loadExample = (type: "itinerary" | "lifeos") => {
    const itineraryExample = `<?xml version='1.0' encoding='utf-8'?>
<Root>
  <Itinerary_Version number="1">
    <patch_info />
    <conversation_history />
    <Day index="1">
        <POI>
          <name>YYZ Airport Arrival</name>
          <time_block>2:00 PM - 3:00 PM</time_block>
          <description>Flight lands at Toronto Pearson International Airport. Long journey from overseas, feeling tired and jet-lagged.</description>
          <travel_method>Flight</travel_method>
          <expected_cost>Included</expected_cost>
          <highlights>
            <highlight>International arrival</highlight>
            <highlight>Customs and baggage</highlight>
          </highlights>
        </POI>
        <POI>
          <name>Harbourfront Stroll</name>
          <time_block>5:00 PM - 6:30 PM</time_block>
          <description>Easy, relaxing walk along the waterfront. Low-energy activity perfect for recovering from travel_method. Beautiful lake views and fresh air.</description>
          <travel_method>Walk</travel_method>
          <expected_cost>Free</expected_cost>
          <highlights>
            <highlight>Lake Ontario views</highlight>
            <highlight>Relaxing atmosphere</highlight>
            <highlight>Outdoor fresh air</highlight>
          </highlights>
        </POI>
        <Restaurant>
          <name>Canoe Restaurant</name>
          <time_block>7:30 PM - 9:00 PM</time_block>
          <description>Upscale Canadian cuisine on the 54th floor of TD Tower. Expensive fine dining with panoramic city views. Elegant and formal atmosphere.</description>
          <travel_method>Taxi</travel_method>
          <expected_cost>CAD 150-200</expected_cost>
          <highlights>
            <highlight>Expensive fine dining</highlight>
            <highlight>City skyline views</highlight>
            <highlight>Upscale atmosphere</highlight>
          </highlights>
        </Restaurant>
    </Day>
  </Itinerary_Version>
</Root>`;

    const lifeOSExample = `<?xml version='1.0' encoding='utf-8'?>
<Root>
  <LifeOS_Version number="1">
    <Category name="Work">
      <description>Research, teaching assistantship, and grant applications.</description>
      <Project>
        <name>NSERC CGS-M Grant Application</name>
        <status>Active</status>
        <deadline>2026-12-01</deadline>
        <priority>High</priority>
        <Task>
          <description>Draft research proposal (max 1 page)</description>
          <status>In Progress</status>
          <priority>High</priority>
          <due_date>2026-11-15</due_date>
        </Task>
        <Task>
          <description>Finalize CCV (Canadian Common CV) updates</description>
          <status>Todo</status>
          <priority>Medium</priority>
          <due_date>2026-11-20</due_date>
        </Task>
      </Project>
    </Category>
    <Category name="Personal">
      <description>Errands, social, and admin.</description>
      <Project>
        <name>General</name>
        <status>Active</status>
        <priority>Low</priority>
        <Task>
          <description>Call Mom regarding weekend plans</description>
          <status>Todo</status>
          <priority>Medium</priority>
          <due_date>2026-02-11</due_date>
        </Task>
      </Project>
    </Category>
  </LifeOS_Version>
</Root>`;

    if (type === "itinerary") onXmlLoaded(itineraryExample);
    if (type === "lifeos") onXmlLoaded(lifeOSExample);
  };

  return (
    <div className="file-input-container">
      <div
        className={`drop-zone ${dragActive ? "active" : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <Upload size={48} className="icon" />
        <p>Drag and drop your XML file here</p>
        <p className="or-text">or</p>
        <label className="file-upload-btn">
          Browse Files
          <input type="file" accept=".xml" onChange={handleChange} />
        </label>
      </div>

      <div className="text-input-section">
        <textarea
          placeholder="Or paste your XML content here..."
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
        />
        <button onClick={handleTextSubmit} disabled={!textInput.trim()}>
          <Code size={16} /> Load XML
        </button>
      </div>

      <div className="example-buttons">
        <button className="text-btn" onClick={() => loadExample("itinerary")}>
          Load Itinerary Example
        </button>
        <button className="text-btn" onClick={() => loadExample("lifeos")}>
          Load LifeOS Example
        </button>
      </div>
    </div>
  );
};
