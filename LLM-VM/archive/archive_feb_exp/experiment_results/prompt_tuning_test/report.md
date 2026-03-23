# Experiment Report: prompt_tuning_test

## Summary: semantic_xpath

| Query | NL Request | Operation | XPath Query | Tokens | Time (s) |
|---|---|---|---|---|---|
| 001 | My friend lives in Mississauga, about an hour from downtown. What days are wi... | READ | `/Itinerary/Day[not(agg_exists(POI[atom(content =~ "work") OR atom(content =~ "flight")]) OR agg_exists(Restaurant[atom(content =~ "work") OR atom(content =~ "flight")]))]` | 6,518 (5,025 / 1,493) | 53.99 |
| 002 | The weather forecast shows heavy rain on Day 7. Which activities are outdoors... | READ | `/Itinerary/Day[7]/POI[atom(content =~ "outdoor") AND atom(content =~ "reschedule")]` | 4,768 (4,132 / 636) | 13.71 |
| 003 | My sister and nephew are joining me on Day 6. He's 10 years old. What activit... | READ | `/Itinerary/Day[6]/.[atom(content =~ "child friendly")]` | 6,150 (4,512 / 1,638) | 24.95 |
| 004 | I'm putting together my expense report. What are the most expensive restauran... | READ | `/Itinerary/Day/Restaurant[atom(content =~ "expensive") AND atom(content =~ "reservation")]` | 5,665 (4,483 / 1,182) | 30.19 |
| 005 | I've been stuck in meetings most of the week. Which days have nature activiti... | READ | `/Itinerary/Day[agg_exists(POI[atom(content =~ "nature") AND atom(content =~ "outdoor")])]` | 7,627 (5,740 / 1,887) | 39.64 |
| 006 | Show me days where I have both outdoor activities and a fine dining dinner pl... | READ | `/Itinerary/Day[agg_exists(POI[atom(content =~ "outdoor")]) AND agg_exists(Restaurant[atom(content =~ "fine dining")])]` | 7,845 (5,630 / 2,215) | 43.53 |
| 007 | List all expensive restaurants that are not serving Italian cuisine. | READ | `/Itinerary/desc::Restaurant[atom(content =~ "expensive") AND not(atom(content =~ "Italian"))]` | 5,498 (4,470 / 1,028) | 26.79 |
| 008 | Which days include a visit to a museum or a gallery, but have absolutely no w... | READ | `/Itinerary/Day[agg_exists(POI[atom(content =~ "museum") OR atom(content =~ "gallery")]) AND not(agg_exists(POI[atom(content =~ "work related")]) OR agg_exists(Restaurant[atom(content =~ "work related")]))]` | 7,452 (5,752 / 1,700) | 47.83 |
| 009 | Identify days where no restaurant is described as expensive or formal. | READ | `/Itinerary/Day[not(agg_exists(Restaurant[atom(content =~ "expensive") OR atom(content =~ "formal")]))]` | 6,134 (4,716 / 1,418) | 34.53 |
| 010 | What free activities can I walk to? | READ | `/Itinerary/Day/POI[atom(content =~ "free") AND atom(content =~ "walk")]` | 5,529 (4,434 / 1,095) | 25.54 |
| 011 | Find restaurants serving French or Japanese cuisine. | READ | `/Itinerary/desc::Restaurant[atom(content =~ "French") OR atom(content =~ "Japanese")]` | 5,237 (4,442 / 795) | 23.72 |


### Stage Breakdown (11 queries)

| Stage | Time (s) | Time % | Prompt | Completion | Total Tokens |
|-------|----------|--------|--------|------------|--------------|
| version_resolution | 41.1s | 11.3% | 10,786 | 2,364 | 13,150 |
| version_lookup | 0.1s | 0.0% | 0 | 0 | 0 |
| query_generation | 79.8s | 21.9% | 24,120 | 5,163 | 29,283 |
| xpath_execution | 137.7s | 37.8% | 0 | 0 | 0 |
| downstream_task | 105.7s | 29.0% | 18,430 | 7,560 | 25,990 |
| **TOTAL** | **364.4s** | **100%** | **53,336** | **15,087** | **68,423** |

**Averages per query:**

| Stage | Avg Time | Avg Tokens |
|-------|----------|------------|
| version_resolution | 3.73s | 1,195 |
| version_lookup | 0.01s | 0 |
| query_generation | 7.25s | 2,662 |
| xpath_execution | 12.52s | 0 |
| downstream_task | 9.61s | 2,363 |
| **TOTAL** | **33.12s** | **6,220** |


## Detailed Results
### Query 001
**Query:** My friend lives in Mississauga, about an hour from downtown. What days are without any work commitments or flights?

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/Day[not(agg_exists(POI[atom(content =~ "work") OR atom(content =~ "flight")]) OR agg_exists(Restaurant[atom(content =~ "work") OR atom(content =~ "flight")]))]`
- **Time:** 53.99s
- **Tokens:** 6,518 (5,025 / 1,493)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | Day 9 | Day 9 lists only restaurants and shopping (brunch, Eaton Centre, boutiques, dinner). No flights or work commitments present. |
| 2 | Day 8 | Day 8 is a leisure Niagara Falls day trip plus restaurants (tour bus transportation). No flights or work commitments present. |
| 3 | Day 7 | Day 7 contains museum visits, walking tours, and restaurants (leisure activities). No flights or work commitments present. |

**Scoring Analysis:**

**Predicate:** `not(agg_exists(POI[atom(content =~ "work") OR atom(content =~ "flight")]) OR agg_exists(Restaurant[atom(content =~ "work") OR atom(content =~ "flight")]))`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | work (POI) | flight (POI) | work (Restaurant) | flight (Restaurant) | Final Score | Result |
|---| --- | --- | --- | --- |---|---|
| Day 1 | 0.7450 | 0.9984 | 0.9298 | 0.0836 | 0.0016 | ❌ Filtered Out |
| Day 2 | 0.9999 | 0.0297 | 0.9464 | 0.2156 | 0.0001 | ❌ Filtered Out |
| Day 3 | 0.9999 | 0.0177 | 0.9726 | 0.0048 | 0.0001 | ❌ Filtered Out |
| Day 4 | 0.9998 | 0.2039 | 0.9989 | 0.0016 | 0.0002 | ❌ Filtered Out |
| Day 5 | 0.9999 | 0.0245 | 0.8115 | 0.2114 | 0.0001 | ❌ Filtered Out |
| Day 6 | 0.9593 | 0.4096 | 0.5445 | 0.0085 | 0.0407 | ❌ Filtered Out |
| Day 7 | 0.9365 | 0.4907 | 0.7189 | 0.0214 | 0.0635 | ✅ Candidate |
| Day 8 | 0.9080 | 0.1027 | 0.8107 | 0.0610 | 0.0920 | ✅ Candidate |
| Day 9 | 0.8030 | 0.0157 | 0.7514 | 0.0096 | 0.1970 | ✅ Candidate |
| Day 10 | 0.9200 | 0.9997 | 0.9967 | 0.0154 | 0.0003 | ❌ Filtered Out |


### Query 002
**Query:** The weather forecast shows heavy rain on Day 7. Which activities are outdoors that I might need to reschedule?

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/Day[7]/POI[atom(content =~ "outdoor") AND atom(content =~ "reschedule")]`
- **Time:** 13.71s
- **Tokens:** 4,768 (4,132 / 636)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | POI: Distillery District Walk | Explicitly described as an outdoor walking experience with cobblestone streets and an outdoor walking tour (3:00 PM - 5:00 PM) — likely affected by... |

**Scoring Analysis:**

**Predicate:** `atom(content =~ "outdoor") AND atom(content =~ "reschedule")`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | outdoor | reschedule | Final Score | Result |
|---| --- | --- |---|---|
| Royal Ontario Museum | 0.1170 | 0.8001 | 0.1170 | ✅ Candidate |
| Distillery District Walk | 0.9981 | 0.7705 | 0.7705 | ✅ Candidate |


### Query 003
**Query:** My sister and nephew are joining me on Day 6. He's 10 years old. What activities do I already have planned that would be fun for a kid?

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/Day[6]/.[atom(content =~ "child friendly")]`
- **Time:** 24.95s
- **Tokens:** 6,150 (4,512 / 1,638)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | POI: Ripley's Aquarium | Ripley's Aquarium is explicitly described as an indoor family attraction with interactive exhibits, touch tanks, and kid-friendly highlights — clea... |
| 2 | POI: Toronto Islands Ferry and Bike Ride | Toronto Islands Ferry and Bike Ride is described as a family- and kid-friendly outdoor activity (ferry ride, bike rental, beaches, gardens) — suita... |
| 3 | Restaurant: Sunset Grill | Sunset Grill is noted as a family-friendly breakfast with a kid-friendly menu and quick service, which makes it appropriate for a 10-year-old joini... |
| 4 | Restaurant: Island Cafe Picnic Lunch | Island Cafe Picnic Lunch is a casual, outdoor picnic on Centre Island with 'family friendly' highlights — a relaxed outdoor meal/activity that a ch... |

**Scoring Analysis:**

**Predicate:** `atom(content =~ "child friendly")`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | child friendly | Final Score | Result |
|---| --- |---|---|
| Sunset Grill | 0.9992 | 0.9992 | ✅ Candidate |
| Toronto Islands Ferry and Bike Ride | 0.9990 | 0.9990 | ✅ Candidate |
| Island Cafe Picnic Lunch | 0.9941 | 0.9941 | ✅ Candidate |
| Ripley's Aquarium | 0.9997 | 0.9997 | ✅ Candidate |
| The Keg Steakhouse | 0.4765 | 0.4765 | ✅ Candidate |


### Query 004
**Query:** I'm putting together my expense report. What are the most expensive restaurants I've booked?

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/Day/Restaurant[atom(content =~ "expensive") AND atom(content =~ "reservation")]`
- **Time:** 30.19s
- **Tokens:** 5,665 (4,483 / 1,182)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | Restaurant: Alo Restaurant | Alo Restaurant — expected_cost CAD 300-400, described as extremely expensive three-Michelin-star tasting menu (highest-cost candidate). |
| 2 | Restaurant: Kaiseki Kaji | Kaiseki Kaji — expected_cost CAD 250-350, described as very expensive kaiseki omakase (second-highest cost). |
| 3 | Restaurant: Canoe Restaurant | Canoe Restaurant — expected_cost CAD 150-200, described as upscale expensive fine dining (next most expensive after the two above). |

**Scoring Analysis:**

**Predicate:** `atom(content =~ "expensive") AND atom(content =~ "reservation")`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | expensive | reservation | Final Score | Result |
|---| --- | --- |---|---|
| Canoe Restaurant | 0.9978 | 0.9503 | 0.9503 | ✅ Candidate |
| Alo Restaurant | 0.9982 | 0.9628 | 0.9628 | ✅ Candidate |
| Hotel Continental Breakfast | 0.1213 | 0.2907 | 0.1213 | ⚪ Above Threshold |
| FRANK Restaurant at AGO | 0.9310 | 0.7921 | 0.7921 | ⚪ Above Threshold |
| Pai Northern Thai | 0.9373 | 0.8875 | 0.8875 | ⚪ Above Threshold |
| Quick Grab Coffee | 0.4355 | 0.0195 | 0.0195 | ❌ Filtered Out |
| Lee Restaurant | 0.9826 | 0.9142 | 0.9142 | ⚪ Above Threshold |
| Mildred's Temple Kitchen | 0.9468 | 0.6483 | 0.6483 | ⚪ Above Threshold |
| Kaiseki Kaji | 0.9987 | 0.9628 | 0.9628 | ✅ Candidate |
| Sunset Grill | 0.3724 | 0.5244 | 0.3724 | ⚪ Above Threshold |
| Island Cafe Picnic Lunch | 0.7898 | 0.2479 | 0.2479 | ⚪ Above Threshold |
| The Keg Steakhouse | 0.7176 | 0.7872 | 0.7176 | ⚪ Above Threshold |
| Fran's Restaurant | 0.5388 | 0.4481 | 0.4481 | ⚪ Above Threshold |
| Eataly Toronto | 0.9085 | 0.6800 | 0.6800 | ⚪ Above Threshold |
| Cluny Bistro | 0.9631 | 0.9290 | 0.9290 | ✅ Candidate |
| Elements on the Falls | 0.9610 | 0.7538 | 0.7538 | ⚪ Above Threshold |
| Antler Kitchen | 0.9850 | 0.9196 | 0.9196 | ✅ Candidate |
| Lady Marmalade | 0.9775 | 0.8155 | 0.8155 | ⚪ Above Threshold |
| Urban Eatery Food Court | 0.1900 | 0.0343 | 0.0343 | ❌ Filtered Out |
| Bar Isabel | 0.8872 | 0.5852 | 0.5852 | ⚪ Above Threshold |
| Hotel Quick Checkout Breakfast | 0.6455 | 0.3867 | 0.3867 | ⚪ Above Threshold |
| 360 Restaurant at CN Tower | 0.9896 | 0.8815 | 0.8815 | ⚪ Above Threshold |


### Query 005
**Query:** I've been stuck in meetings most of the week. Which days have nature activities where I can actually get outside?

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/Day[agg_exists(POI[atom(content =~ "nature") AND atom(content =~ "outdoor")])]`
- **Time:** 39.64s
- **Tokens:** 7,627 (5,740 / 1,887)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | Day 8 | Day 8 includes a Niagara Falls Day Trip — a full-day outdoor scenic adventure (falls, boat ride, time to explore) — clearly a nature/outdoor activity. |
| 2 | Day 6 | Day 6 includes the Toronto Islands Ferry and Bike Ride — ferry to islands, bike exploration, beaches and gardens — explicitly an outdoor nature act... |
| 3 | Day 1 | Day 1 includes a Harbourfront Stroll — an easy outdoor waterfront walk with fresh air and lake views. |

**Scoring Analysis:**

**Predicate:** `agg_exists(POI[atom(content =~ "nature") AND atom(content =~ "outdoor")])`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | nature (POI) | outdoor (POI) | Final Score | Result |
|---| --- | --- |---|---|
| Day 1 | 0.9874 | 0.9976 | 0.9874 | ✅ Candidate |
| Day 2 | 0.2128 | 0.6112 | 0.2128 | ⚪ Above Threshold |
| Day 3 | 0.1042 | 0.8928 | 0.1042 | ⚪ Above Threshold |
| Day 4 | 0.6491 | 0.9698 | 0.6491 | ⚪ Above Threshold |
| Day 5 | 0.0431 | 0.0902 | 0.0431 | ❌ Filtered Out |
| Day 6 | 0.9992 | 0.9992 | 0.9992 | ✅ Candidate |
| Day 7 | 0.5300 | 0.9981 | 0.5300 | ⚪ Above Threshold |
| Day 8 | 0.9994 | 0.9995 | 0.9994 | ✅ Candidate |
| Day 9 | 0.9455 | 0.9987 | 0.9455 | ✅ Candidate |
| Day 10 | 0.8239 | 0.9965 | 0.8239 | ✅ Candidate |


### Query 006
**Query:** Show me days where I have both outdoor activities and a fine dining dinner planned.

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/Day[agg_exists(POI[atom(content =~ "outdoor")]) AND agg_exists(Restaurant[atom(content =~ "fine dining")])]`
- **Time:** 43.53s
- **Tokens:** 7,845 (5,630 / 2,215)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | Day 1 | Day 1 has an explicit outdoor activity (Harbourfront Stroll: waterfront walk, outdoor fresh air) and an evening upscale/fine dining restaurant (Can... |
| 2 | Day 8 | Day 8 includes a clearly outdoor activity (Niagara Falls Day Trip: full-day outdoor scenic waterfall excursion) and an evening dinner (Antler Kitch... |
| 3 | Day 7 | Day 7 features an outdoor activity (Distillery District Walk: outdoor walking experience through historic pedestrian village) and an evening dinner... |

**Scoring Analysis:**

**Predicate:** `agg_exists(POI[atom(content =~ "outdoor")]) AND agg_exists(Restaurant[atom(content =~ "fine dining")])`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | outdoor (POI) | fine dining (Restaurant) | Final Score | Result |
|---| --- | --- |---|---|
| Day 1 | 0.9976 | 0.9986 | 0.9976 | ✅ Candidate |
| Day 2 | 0.6113 | 0.9995 | 0.6113 | ⚪ Above Threshold |
| Day 3 | 0.8928 | 0.9714 | 0.8928 | ⚪ Above Threshold |
| Day 4 | 0.9698 | 0.9903 | 0.9698 | ✅ Candidate |
| Day 5 | 0.1775 | 0.9986 | 0.1775 | ⚪ Above Threshold |
| Day 6 | 0.9992 | 0.9491 | 0.9491 | ⚪ Above Threshold |
| Day 7 | 0.9981 | 0.9789 | 0.9789 | ✅ Candidate |
| Day 8 | 0.9995 | 0.9951 | 0.9951 | ✅ Candidate |
| Day 9 | 0.9987 | 0.2031 | 0.2031 | ⚪ Above Threshold |
| Day 10 | 0.9965 | 0.9935 | 0.9935 | ✅ Candidate |


### Query 007
**Query:** List all expensive restaurants that are not serving Italian cuisine.

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/desc::Restaurant[atom(content =~ "expensive") AND not(atom(content =~ "Italian"))]`
- **Time:** 26.79s
- **Tokens:** 5,498 (4,470 / 1,028)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | Restaurant: Kaiseki Kaji | Kaiseki Kaji is explicitly described as 'Very expensive omakase' and 'Expensive Japanese dining' with expected_cost CAD 250-350 — expensive and not... |
| 2 | Restaurant: Canoe Restaurant | Canoe Restaurant is described as 'Expensive fine dining' (expected_cost CAD 150-200) serving Canadian cuisine — expensive and not Italian. |

**Scoring Analysis:**

**Predicate:** `atom(content =~ "expensive") AND not(atom(content =~ "Italian"))`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | expensive | Italian | Final Score | Result |
|---| --- | --- |---|---|
| Canoe Restaurant | 0.9978 | 0.0023 | 0.9977 | ✅ Candidate |
| Alo Restaurant | 0.9982 | 0.0889 | 0.9111 | ⚪ Above Threshold |
| Hotel Continental Breakfast | 0.1213 | 0.0237 | 0.1213 | ⚪ Above Threshold |
| FRANK Restaurant at AGO | 0.9310 | 0.0266 | 0.9310 | ⚪ Above Threshold |
| Pai Northern Thai | 0.9373 | 0.0004 | 0.9373 | ⚪ Above Threshold |
| Quick Grab Coffee | 0.4355 | 0.0021 | 0.4355 | ⚪ Above Threshold |
| Lee Restaurant | 0.9826 | 0.0016 | 0.9826 | ✅ Candidate |
| Mildred's Temple Kitchen | 0.9468 | 0.3465 | 0.6535 | ⚪ Above Threshold |
| Kaiseki Kaji | 0.9987 | 0.0023 | 0.9977 | ✅ Candidate |
| Sunset Grill | 0.3724 | 0.0752 | 0.3724 | ⚪ Above Threshold |
| Island Cafe Picnic Lunch | 0.7898 | 0.3885 | 0.6115 | ⚪ Above Threshold |
| The Keg Steakhouse | 0.7176 | 0.0207 | 0.7176 | ⚪ Above Threshold |
| Fran's Restaurant | 0.5388 | 0.6325 | 0.3675 | ⚪ Above Threshold |
| Eataly Toronto | 0.9085 | 0.9996 | 0.0004 | ❌ Filtered Out |
| Cluny Bistro | 0.9631 | 0.0012 | 0.9631 | ⚪ Above Threshold |
| Elements on the Falls | 0.9610 | 0.2370 | 0.7630 | ⚪ Above Threshold |
| Antler Kitchen | 0.9850 | 0.0132 | 0.9850 | ✅ Candidate |
| Lady Marmalade | 0.9775 | 0.1246 | 0.8754 | ⚪ Above Threshold |
| Urban Eatery Food Court | 0.1900 | 0.2447 | 0.1900 | ⚪ Above Threshold |
| Bar Isabel | 0.8872 | 0.0006 | 0.8872 | ⚪ Above Threshold |
| Hotel Quick Checkout Breakfast | 0.6455 | 0.2544 | 0.6455 | ⚪ Above Threshold |
| 360 Restaurant at CN Tower | 0.9896 | 0.0134 | 0.9866 | ✅ Candidate |


### Query 008
**Query:** Which days include a visit to a museum or a gallery, but have absolutely no work-related commitments?

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/Day[agg_exists(POI[atom(content =~ "museum") OR atom(content =~ "gallery")]) AND not(agg_exists(POI[atom(content =~ "work related")]) OR agg_exists(Restaurant[atom(content =~ "work related")]))]`
- **Time:** 47.83s
- **Tokens:** 7,452 (5,752 / 1,700)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | Day 7 | Day 7 contains the Royal Ontario Museum (explicit museum) and the Distillery District walk which mentions galleries. The day's subtree contains no ... |

**Scoring Analysis:**

**Predicate:** `agg_exists(POI[atom(content =~ "museum") OR atom(content =~ "gallery")]) AND not(agg_exists(POI[atom(content =~ "work related")]) OR agg_exists(Restaurant[atom(content =~ "work related")]))`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | museum (POI) | gallery (POI) | work related (POI) | work related (Restaurant) | Final Score | Result |
|---| --- | --- | --- | --- |---|---|
| Day 1 | 0.0202 | 0.8073 | 0.9360 | 0.7270 | 0.0640 | ✅ Candidate |
| Day 2 | 0.1408 | 0.1072 | 0.9999 | 0.6939 | 0.0001 | ❌ Filtered Out |
| Day 3 | 0.9977 | 0.9942 | 0.9998 | 0.8624 | 0.0002 | ❌ Filtered Out |
| Day 4 | 0.6247 | 0.7738 | 0.9998 | 0.9992 | 0.0002 | ❌ Filtered Out |
| Day 5 | 0.0501 | 0.2212 | 0.9999 | 0.4962 | 0.0001 | ❌ Filtered Out |
| Day 6 | 0.7643 | 0.9921 | 0.6812 | 0.1619 | 0.3188 | ✅ Candidate |
| Day 7 | 0.9981 | 0.9308 | 0.8614 | 0.3700 | 0.1386 | ✅ Candidate |
| Day 8 | 0.0415 | 0.9881 | 0.7503 | 0.3810 | 0.2497 | ✅ Candidate |
| Day 9 | 0.0094 | 0.8814 | 0.3824 | 0.3722 | 0.6176 | ✅ Candidate |
| Day 10 | 0.1051 | 0.9632 | 0.8617 | 0.9978 | 0.0022 | ❌ Filtered Out |


### Query 009
**Query:** Identify days where no restaurant is described as expensive or formal.

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/Day[not(agg_exists(Restaurant[atom(content =~ "expensive") OR atom(content =~ "formal")]))]`
- **Time:** 34.53s
- **Tokens:** 6,134 (4,716 / 1,418)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | Day 3 | All restaurants on Day 3 are described as simple, convenient, artistic, or lively with moderate expected_costs (CAD 25–50). None are described as e... |

**Scoring Analysis:**

**Predicate:** `not(agg_exists(Restaurant[atom(content =~ "expensive") OR atom(content =~ "formal")]))`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | expensive (Restaurant) | formal (Restaurant) | Final Score | Result |
|---| --- | --- |---|---|
| Day 1 | 0.9978 | 0.9947 | 0.0022 | ❌ Filtered Out |
| Day 2 | 0.9982 | 0.9991 | 0.0009 | ❌ Filtered Out |
| Day 3 | 0.9373 | 0.0397 | 0.0627 | ✅ Candidate |
| Day 4 | 0.9826 | 0.1088 | 0.0174 | ❌ Filtered Out |
| Day 5 | 0.9987 | 0.9990 | 0.0010 | ❌ Filtered Out |
| Day 6 | 0.7898 | 0.0120 | 0.2102 | ✅ Candidate |
| Day 7 | 0.9631 | 0.1207 | 0.0369 | ❌ Filtered Out |
| Day 8 | 0.9850 | 0.2134 | 0.0150 | ❌ Filtered Out |
| Day 9 | 0.9775 | 0.0195 | 0.0225 | ❌ Filtered Out |
| Day 10 | 0.9896 | 0.2898 | 0.0104 | ❌ Filtered Out |


### Query 010
**Query:** What free activities can I walk to?

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/Day/POI[atom(content =~ "free") AND atom(content =~ "walk")]`
- **Time:** 25.54s
- **Tokens:** 5,529 (4,434 / 1,095)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | POI: Harbourfront Stroll | Harbourfront Stroll is explicitly marked expected_cost: Free and travel_method: Walk, so it directly matches 'free activities I can walk to.' |

**Scoring Analysis:**

**Predicate:** `atom(content =~ "free") AND atom(content =~ "walk")`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | free | walk | Final Score | Result |
|---| --- | --- |---|---|
| YYZ Airport Arrival | 0.2932 | 0.0028 | 0.0028 | ❌ Filtered Out |
| Harbourfront Stroll | 0.9987 | 0.9991 | 0.9987 | ✅ Candidate |
| Client Kickoff Meeting | 0.0230 | 0.9961 | 0.0230 | ❌ Filtered Out |
| Strategy Workshop | 0.0786 | 0.9968 | 0.0786 | ⚪ Above Threshold |
| Stakeholder Presentation | 0.0076 | 0.9967 | 0.0076 | ❌ Filtered Out |
| Art Gallery of Ontario | 0.2126 | 0.9993 | 0.2126 | ⚪ Above Threshold |
| Full-Day Workshop | 0.1046 | 0.9985 | 0.1046 | ⚪ Above Threshold |
| Bloor Street Shopping | 0.7468 | 0.9995 | 0.7468 | ✅ Candidate |
| Client Check-in Call | 0.2483 | 0.0079 | 0.0079 | ❌ Filtered Out |
| Email Catch-up Block | 0.0037 | 0.0025 | 0.0025 | ❌ Filtered Out |
| Networking Drinks | 0.1140 | 0.9939 | 0.1140 | ⚪ Above Threshold |
| Toronto Islands Ferry and Bike Ride | 0.2372 | 0.5928 | 0.2372 | ⚪ Above Threshold |
| Ripley's Aquarium | 0.1887 | 0.9772 | 0.1887 | ⚪ Above Threshold |
| Royal Ontario Museum | 0.1488 | 0.9988 | 0.1488 | ⚪ Above Threshold |
| Distillery District Walk | 0.9976 | 0.9932 | 0.9932 | ✅ Candidate |
| Niagara Falls Day Trip | 0.2513 | 0.1316 | 0.1316 | ⚪ Above Threshold |
| Eaton Centre Shopping | 0.2273 | 0.0584 | 0.0584 | ⚪ Above Threshold |
| Queen Street West Boutiques | 0.5796 | 0.9989 | 0.5796 | ✅ Candidate |
| CN Tower EdgeWalk | 0.0633 | 0.9991 | 0.0633 | ⚪ Above Threshold |
| Airport Snack Stop | 0.7034 | 0.9987 | 0.7034 | ✅ Candidate |
| YYZ Departure | 0.4273 | 0.0054 | 0.0054 | ❌ Filtered Out |


### Query 011
**Query:** Find restaurants serving French or Japanese cuisine.

#### semantic_xpath
- **Operation:** READ
- **XPath:** `/Itinerary/desc::Restaurant[atom(content =~ "French") OR atom(content =~ "Japanese")]`
- **Time:** 23.72s
- **Tokens:** 5,237 (4,442 / 795)

**Selected Nodes:**

| # | Node | Reasoning |
|---|------|-----------|
| 1 | Restaurant: Kaiseki Kaji | Kaiseki Kaji is explicitly Japanese (kaiseki omakase dining). |
| 2 | Restaurant: Cluny Bistro | Cluny Bistro is explicitly French-inspired (French bistro). |
| 3 | Restaurant: Alo Restaurant | Alo Restaurant offers a French tasting menu (three Michelin-starred French cuisine). |

**Scoring Analysis:**

**Predicate:** `atom(content =~ "French") OR atom(content =~ "Japanese")`
**Threshold:** `0.05` | **Top-K:** `5`

| Node | French | Japanese | Final Score | Result |
|---| --- | --- |---|---|
| Canoe Restaurant | 0.3134 | 0.0055 | 0.3134 | ⚪ Above Threshold |
| Alo Restaurant | 0.9989 | 0.0894 | 0.9989 | ✅ Candidate |
| Hotel Continental Breakfast | 0.9029 | 0.0336 | 0.9029 | ⚪ Above Threshold |
| FRANK Restaurant at AGO | 0.9870 | 0.0036 | 0.9870 | ⚪ Above Threshold |
| Pai Northern Thai | 0.0013 | 0.0051 | 0.0051 | ❌ Filtered Out |
| Quick Grab Coffee | 0.5471 | 0.0078 | 0.5471 | ⚪ Above Threshold |
| Lee Restaurant | 0.0074 | 0.2846 | 0.2846 | ⚪ Above Threshold |
| Mildred's Temple Kitchen | 0.8438 | 0.3693 | 0.8438 | ⚪ Above Threshold |
| Kaiseki Kaji | 0.0570 | 0.9996 | 0.9996 | ✅ Candidate |
| Sunset Grill | 0.8809 | 0.4672 | 0.8809 | ⚪ Above Threshold |
| Island Cafe Picnic Lunch | 0.9102 | 0.3463 | 0.9102 | ⚪ Above Threshold |
| The Keg Steakhouse | 0.8153 | 0.0449 | 0.8153 | ⚪ Above Threshold |
| Fran's Restaurant | 0.9924 | 0.0078 | 0.9924 | ✅ Candidate |
| Eataly Toronto | 0.0190 | 0.0038 | 0.0190 | ❌ Filtered Out |
| Cluny Bistro | 0.9992 | 0.0018 | 0.9992 | ✅ Candidate |
| Elements on the Falls | 0.7604 | 0.1426 | 0.7604 | ⚪ Above Threshold |
| Antler Kitchen | 0.8359 | 0.0066 | 0.8359 | ⚪ Above Threshold |
| Lady Marmalade | 0.9885 | 0.0772 | 0.9885 | ✅ Candidate |
| Urban Eatery Food Court | 0.5357 | 0.1390 | 0.5357 | ⚪ Above Threshold |
| Bar Isabel | 0.0059 | 0.0012 | 0.0059 | ❌ Filtered Out |
| Hotel Quick Checkout Breakfast | 0.9647 | 0.6023 | 0.9647 | ⚪ Above Threshold |
| 360 Restaurant at CN Tower | 0.5126 | 0.0712 | 0.5126 | ⚪ Above Threshold |

