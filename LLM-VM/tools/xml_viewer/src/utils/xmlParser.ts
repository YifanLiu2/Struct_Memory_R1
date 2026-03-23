import { XMLParser } from "fast-xml-parser";

export type ViewType = "itinerary" | "todolist" | "mealplan" | "generic";

export interface ParsedDocument {
  type: ViewType;
  data: any;
  label: string;
}

export type ParsedData = ParsedDocument[];

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "@_",
});

export const ensureArray = <T>(item: T | T[] | undefined): T[] => {
  if (!item) return [];
  if (Array.isArray(item)) return item;
  return [item];
};

export const parseXML = (xmlContent: string): ParsedData => {
  try {
    const jsonObj = parser.parse(xmlContent);
    const root = jsonObj.Root || jsonObj;
    const documents: ParsedData = [];

    // Handle Itineraries
    if (root.Itinerary_Version) {
      const itineraries = ensureArray(root.Itinerary_Version);
      itineraries.forEach((data: any, index: number) => {
        const version = data["@_number"] || index + 1;
        documents.push({
          type: "itinerary",
          data: data,
          label: `Itinerary V${version}`,
        });
      });
    }

    // Handle ToDoLists (formerly LifeOS)
    if (root.ToDoList_Version) {
      const todolists = ensureArray(root.ToDoList_Version);
      todolists.forEach((data: any, index: number) => {
        const version = data["@_number"] || index + 1;
        documents.push({
          type: "todolist",
          data: data,
          label: `ToDo List V${version}`,
        });
      });
    }

    // Handle MealPlans
    if (root.MealPlan_Version) {
      const mealplans = ensureArray(root.MealPlan_Version);
      mealplans.forEach((data: any, index: number) => {
        const version = data["@_number"] || index + 1;
        documents.push({
          type: "mealplan",
          data: data,
          label: `Meal Plan V${version}`,
        });
      });
    }

    // Legacy/Alternative check for LifeOS if user still uses old tag
    if (root.LifeOS_Version) {
      const lifeOS = ensureArray(root.LifeOS_Version);
      lifeOS.forEach((data: any, index: number) => {
        const version = data["@_number"] || index + 1;
        documents.push({
          type: "todolist",
          data: data,
          label: `LifeOS V${version}`, // Use LifeOS label if tag is LifeOS_Version
        });
      });
    }

    // If no specific documents found, treat as generic
    if (documents.length === 0) {
      documents.push({
        type: "generic",
        data: jsonObj,
        label: "XML View",
      });
    }

    return documents;
  } catch (error) {
    console.error("XML Parsing Error", error);
    return [];
  }
};
