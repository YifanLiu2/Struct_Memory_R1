export interface Highlight {
  highlight: string[];
}

export interface Activity {
  name: string;
  time_block: string;
  description: string;
  travel_method: string;
  expected_cost: string;
  highlights?: Highlight;
}

export interface Day {
  "@_index": string;
  POI?: Activity[] | Activity;
  Restaurant?: Activity[] | Activity;
}

export interface ItineraryData {
  "@_number"?: string;
  Day: Day[];
}

export interface Task {
  description: string;
  status: string;
  priority: string;
  due_date: string;
  note?: string;
}

export interface Project {
  name: string;
  status: string;
  deadline?: string;
  priority: string;
  Task?: Task[] | Task;
}

export interface Category {
  "@_name": string;
  description: string;
  Project?: Project[] | Project;
}

export interface ToDoListData {
  "@_number"?: string;
  patch_info?: string;
  conversation_history?: string;
  Category: Category[];
}

/* ─── MealPlan Types ─── */

export interface Ingredient {
  "@_name": string;
  "@_unit": string;
  "@_quantity": string;
}

export interface Nutrition {
  calories_kcal: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  fiber_g: number;
  sodium_mg: number;
}

export interface MealPlanMeal {
  name: string;
  short_description: string;
  time_block: string;
  meal_suggestion: string;
  ingredients: { item: Ingredient | Ingredient[] };
  cook_time_minutes: number;
  estimated_cost_cad: string;
  nutrition: Nutrition;
}

export interface MealPlanData {
  "@_number"?: string;
  patch_info?: string;
  conversation_history?: string;
  Person: any | any[];
}
