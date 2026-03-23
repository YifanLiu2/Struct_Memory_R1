import React, { useState } from "react";
import {
  UtensilsCrossed,
  Clock,
  DollarSign,
  Flame,
  Beef,
  Wheat,
  Droplets,
  Leaf,
  ChefHat,
  Sun,
  Sunset,
  Moon,
  User,
  CalendarDays,
  ChevronDown,
  ChevronRight,
  ShoppingBasket,
  MessageSquare,
  GitBranch,
  Lightbulb,
  Heart,
  CookingPot,
} from "lucide-react";
import type { MealPlanData, MealPlanMeal, Ingredient } from "../types";
import { ensureArray } from "../utils/xmlParser";

interface MealPlanViewProps {
  data: MealPlanData;
}

/* ─── Version Meta banner (reused pattern) ─── */
const VersionMeta: React.FC<{ data: MealPlanData }> = ({ data }) => {
  const conversationHistory = data.conversation_history;
  const patchInfo = data.patch_info;

  if (!conversationHistory && !patchInfo) return null;

  const hasConversation =
    conversationHistory &&
    typeof conversationHistory === "string" &&
    conversationHistory.trim().length > 0;
  const hasPatch =
    patchInfo && typeof patchInfo === "string" && patchInfo.trim().length > 0;

  if (!hasConversation && !hasPatch) return null;

  const parsePatchInfo = (text: string) => {
    const prefixMatch = text.match(
      /^((?:Deleted|Updated|Created|Modified)(?:\s+\d+\s+nodes?)?)\s*:\s*/i,
    );
    if (!prefixMatch) return { prefix: null, paths: [text] };
    const prefix = prefixMatch[1];
    const remainder = text.slice(prefixMatch[0].length);
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

/* ─── Nutrition bar/chip ─── */
const NutritionBar: React.FC<{ meal: MealPlanMeal }> = ({ meal }) => {
  const n = meal.nutrition;
  if (!n) return null;

  const items = [
    {
      label: "Cal",
      value: n.calories_kcal,
      unit: "kcal",
      color: "#ef4444",
      icon: <Flame size={13} />,
    },
    {
      label: "Protein",
      value: n.protein_g,
      unit: "g",
      color: "#3b82f6",
      icon: <Beef size={13} />,
    },
    {
      label: "Carbs",
      value: n.carbs_g,
      unit: "g",
      color: "#f59e0b",
      icon: <Wheat size={13} />,
    },
    {
      label: "Fat",
      value: n.fat_g,
      unit: "g",
      color: "#8b5cf6",
      icon: <Droplets size={13} />,
    },
    {
      label: "Fiber",
      value: n.fiber_g,
      unit: "g",
      color: "#22c55e",
      icon: <Leaf size={13} />,
    },
  ];

  return (
    <div className="mp-nutrition-bar">
      {items.map((it) => (
        <div
          key={it.label}
          className="mp-nutrition-chip"
          style={{ "--chip-accent": it.color } as React.CSSProperties}
        >
          <span className="mp-nutrition-chip-icon">{it.icon}</span>
          <span className="mp-nutrition-chip-value">{it.value}</span>
          <span className="mp-nutrition-chip-label">
            {it.unit === "kcal" ? "kcal" : `g ${it.label.toLowerCase()}`}
          </span>
        </div>
      ))}
    </div>
  );
};

/* ─── Ingredient list ─── */
const IngredientList: React.FC<{ ingredients: Ingredient[] }> = ({
  ingredients,
}) => {
  if (!ingredients || ingredients.length === 0) return null;

  return (
    <div className="mp-ingredients">
      <div className="mp-ingredients-header">
        <ShoppingBasket size={14} />
        <span>Ingredients</span>
        <span className="mp-ingredients-count">{ingredients.length}</span>
      </div>
      <div className="mp-ingredients-grid">
        {ingredients.map((item, idx) => {
          const hasQty = item["@_quantity"] && item["@_unit"];
          return (
            <span key={idx} className="mp-ingredient-tag">
              <span className="mp-ingredient-dot" />
              {item["@_name"]}
              {hasQty && (
                <span className="mp-ingredient-qty">
                  {item["@_quantity"]} {item["@_unit"]}
                </span>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
};

/* ─── Nutrition Note ─── */
const NutritionNote: React.FC<{ note: string }> = ({ note }) => {
  if (!note || !note.trim()) return null;

  return (
    <div className="mp-nutrition-note">
      <div className="mp-nutrition-note-header">
        <Heart size={14} />
        <span>Nutrition Notes</span>
      </div>
      <p className="mp-nutrition-note-text">{note}</p>
    </div>
  );
};

/* ─── Meal Suggestion (recipe-style) ─── */
const MealSuggestion: React.FC<{ suggestion: string }> = ({ suggestion }) => {
  if (!suggestion || !suggestion.trim()) return null;

  return (
    <div className="mp-recipe-suggestion">
      <div className="mp-recipe-suggestion-header">
        <CookingPot size={14} />
        <span>How to Prepare</span>
      </div>
      <p className="mp-recipe-suggestion-text">{suggestion}</p>
    </div>
  );
};

/* ─── Single Meal Card ─── */
const MealCard: React.FC<{ meal: MealPlanMeal; index: number }> = ({
  meal,
  index,
}) => {
  const [expanded, setExpanded] = useState(false);
  const ingredients = meal.ingredients
    ? ensureArray(meal.ingredients.item)
    : [];

  // Detect if this is a text-only meal (no numerical nutrition data)
  const hasNumericalNutrition = meal.nutrition && meal.nutrition.calories_kcal;
  const nutritionNote = (meal as any).nutrition_note;
  const hasTextNutrition = nutritionNote && typeof nutritionNote === "string" && nutritionNote.trim();

  return (
    <div className="mp-meal-card">
      <div
        className="mp-meal-card-header"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="mp-meal-card-title-row">
          <span className="mp-meal-option-number">Option {index + 1}</span>
          <h4 className="mp-meal-name">{meal.name}</h4>
          <span className="mp-expand-toggle">
            {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </span>
        </div>

        {meal.short_description && (
          <p className="mp-meal-short-desc">{meal.short_description}</p>
        )}

        <div className="mp-meal-quick-stats">
          {ingredients.length > 0 && (
            <div className="mp-quick-stat">
              <Lightbulb size={13} />
              <span>{ingredients.length} ingredients</span>
            </div>
          )}
          {meal.time_block && (
            <div className="mp-quick-stat">
              <Clock size={13} />
              <span>{meal.time_block}</span>
            </div>
          )}
          {meal.cook_time_minutes && (
            <div className="mp-quick-stat">
              <ChefHat size={13} />
              <span>{meal.cook_time_minutes} min</span>
            </div>
          )}
          {meal.estimated_cost_cad && (
            <div className="mp-quick-stat">
              <DollarSign size={13} />
              <span>{meal.estimated_cost_cad}</span>
            </div>
          )}
        </div>
      </div>

      {expanded && (
        <div className="mp-meal-card-body">
          <IngredientList ingredients={ingredients} />
          <MealSuggestion suggestion={meal.meal_suggestion} />
          {hasNumericalNutrition && <NutritionBar meal={meal} />}
          {hasTextNutrition && <NutritionNote note={nutritionNote} />}
        </div>
      )}
    </div>
  );
};

/* ─── Meal Type Section (Breakfast / Lunch / Dinner) ─── */
type MealType = "Breakfast" | "Lunch" | "Dinner";

const mealTypeConfig: Record<
  MealType,
  { icon: React.ReactNode; gradient: string }
> = {
  Breakfast: {
    icon: <Sun size={18} />,
    gradient: "linear-gradient(135deg, #fbbf24, #f59e0b)",
  },
  Lunch: {
    icon: <Sunset size={18} />,
    gradient: "linear-gradient(135deg, #34d399, #10b981)",
  },
  Dinner: {
    icon: <Moon size={18} />,
    gradient: "linear-gradient(135deg, #818cf8, #6366f1)",
  },
};

const MealTypeSection: React.FC<{
  type: MealType;
  meals: MealPlanMeal[];
}> = ({ type, meals }) => {
  const config = mealTypeConfig[type];
  if (!meals || meals.length === 0) return null;

  return (
    <div className={`mp-meal-type-section mp-${type.toLowerCase()}`}>
      <div className="mp-meal-type-header">
        <span
          className="mp-meal-type-icon"
          style={{ background: config.gradient }}
        >
          {config.icon}
        </span>
        <h3 className="mp-meal-type-title">{type}</h3>
        <span className="mp-meal-type-count">
          {meals.length} option{meals.length !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="mp-meal-cards">
        {meals.map((meal, idx) => (
          <MealCard key={idx} meal={meal} index={idx} />
        ))}
      </div>
    </div>
  );
};

/* ─── Day Section ─── */
const DaySection: React.FC<{ day: any }> = ({ day }) => {
  const dayIndex = day["@_index"] || "?";
  const breakfast = day.Breakfast
    ? ensureArray(day.Breakfast.Meal || day.Breakfast)
    : [];
  const lunch = day.Lunch ? ensureArray(day.Lunch.Meal || day.Lunch) : [];
  const dinner = day.Dinner ? ensureArray(day.Dinner.Meal || day.Dinner) : [];

  return (
    <div className="mp-day-section">
      <div className="mp-day-header">
        <CalendarDays size={20} />
        <h2 className="mp-day-title">Day {dayIndex}</h2>
      </div>
      <div className="mp-day-content">
        <MealTypeSection type="Breakfast" meals={breakfast} />
        <MealTypeSection type="Lunch" meals={lunch} />
        <MealTypeSection type="Dinner" meals={dinner} />
      </div>
    </div>
  );
};

/* ─── Person Section ─── */
const PersonSection: React.FC<{ person: any }> = ({ person }) => {
  const personName = person["@_name"] || "Person";
  const days = ensureArray(person.Day);

  return (
    <div className="mp-person-section">
      <div className="mp-person-header">
        <User size={22} />
        <h2 className="mp-person-name">{personName}</h2>
        <span className="mp-person-day-count">
          {days.length} day{days.length !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="mp-person-days">
        {days.map((day: any, idx: number) => (
          <DaySection key={idx} day={day} />
        ))}
      </div>
    </div>
  );
};

/* ─── Main View ─── */
export const MealPlanView: React.FC<MealPlanViewProps> = ({ data }) => {
  const persons = ensureArray(data.Person);

  return (
    <div className="mealplan-view">
      <h1 className="view-title">
        <UtensilsCrossed size={28} />
        Meal Plan{data["@_number"] ? ` (v${data["@_number"]})` : ""}
      </h1>
      <VersionMeta data={data} />
      {persons.map((person: any, idx: number) => (
        <PersonSection key={idx} person={person} />
      ))}
    </div>
  );
};
