import React from "react";
import {
  MapPin,
  Utensils,
  Clock,
  DollarSign,
  Navigation,
  Star,
} from "lucide-react";
import type { ItineraryData, Activity } from "../types";
import { ensureArray } from "../utils/xmlParser";

interface ItineraryViewProps {
  data: ItineraryData;
}

const ActivityCard: React.FC<{
  activity: Activity;
  type: "POI" | "Restaurant";
}> = ({ activity, type }) => {
  const isRestaurant = type === "Restaurant";
  const Icon = isRestaurant ? Utensils : MapPin;

  return (
    <div className={`activity-card ${type.toLowerCase()}`}>
      <div className="activity-header">
        <div className="activity-title-group">
          <Icon className="activity-icon" size={20} />
          <h3 className="activity-title">{activity.name}</h3>
        </div>
        <div className="activity-time">
          <Clock size={16} />
          <span>{activity.time_block}</span>
        </div>
      </div>

      <p className="activity-description">{activity.description}</p>

      <div className="activity-meta">
        <div className="meta-item">
          <Navigation size={16} />
          <span>{activity.travel_method}</span>
        </div>
        <div className="meta-item">
          <DollarSign size={16} />
          <span>{activity.expected_cost}</span>
        </div>
      </div>

      {activity.highlights && (
        <div className="activity-highlights">
          {ensureArray(activity.highlights.highlight).map((highlight, idx) => (
            <span key={idx} className="highlight-tag">
              <Star size={12} /> {highlight}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

export const ItineraryView: React.FC<ItineraryViewProps> = ({ data }) => {
  const days = ensureArray(data.Day);

  return (
    <div className="itinerary-view">
      <h1 className="view-title">Travel Itinerary</h1>
      <div className="timeline-container">
        {days.map((day) => (
          <div key={day["@_index"]} className="timeline-day">
            <div className="day-marker">
              <span className="day-label">Day {day["@_index"]}</span>
            </div>
            <div className="day-content">
              {ensureArray(day.POI).map((poi, idx) => (
                <ActivityCard key={`poi-${idx}`} activity={poi} type="POI" />
              ))}
              {ensureArray(day.Restaurant).map((restaurant, idx) => (
                <ActivityCard
                  key={`rest-${idx}`}
                  activity={restaurant}
                  type="Restaurant"
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
