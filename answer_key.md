# Answer Key — Planted Problems in the Gulftrade Dataset

All scenarios below are deliberately embedded in the generated CSV files.
Use this as the ground-truth checklist for evaluating hackathon solutions.

---

## 1. Hero Case — USB-C Charger stock/demand imbalance (SKU-001)

**Location:** `inventory.csv` + `sales_history.csv`

| Location   | Available Stock | ~30-Day Demand | Surplus / (Shortfall) |
|------------|-----------------|----------------|-----------------------|
| Dubai      | 500             | ~210           | +290 (overstocked)    |
| Abu Dhabi  | 100             | ~360           | −260 (understocked)   |
| Sharjah    | 300             | ~150           | +150 (overstocked)    |

**The intended action:** Transfer stock from Dubai (and possibly Sharjah)
into Abu Dhabi to prevent a stockout. SKU-001 is a USB-C charger with the
highest demand volume, making it the "hero" optimization opportunity.

Daily demand baselines used: Dubai 7, Abu Dhabi 12, Sharjah 5.

---

## 2. 10-Day Stockout in Abu Dhabi (SKU-007)

**Location:** `sales_history.csv`

- **SKU:** SKU-007 (FlexLink Lightning 1m cable)
- **Location:** Abu Dhabi
- **Window:** Days 78–87 of the 90-day range (2026-10-15 → 2026-10-24, 10 consecutive days, ending 3 days before the as-of date 2026-10-27)
- **Symptom:** `units_sold` drops to exactly **0** for those 10 days.

A correct analysis should detect this zero-sales streak as a stockout,
not as "no demand."

---

## 3. 8-Day Stockout in Dubai (SKU-013)

**Location:** `sales_history.csv`

- **SKU:** SKU-013 (TypeFlow Mini keyboard)
- **Location:** Dubai
- **Window:** Days 43–50 of the 90-day range (2026-09-10 → 2026-09-17, 8 consecutive days)
- **Symptom:** `units_sold` drops to exactly **0** for those 8 days.

This stockout falls in the **training period** of the backtest (first 60
days), so the smart forecast should detect it, exclude the zero-sales days,
and fill them with the average demand — improving accuracy vs. the naive
baseline that treats the zeros as real demand.

---

## 4. Missing Sales Records

**Location:** `sales_history.csv` (rows are simply absent)

| SKU      | Location   | Day index | Approx. date |
|----------|------------|-----------|--------------|
| SKU-003  | Sharjah    | 10        | 2026-08-09   |
| SKU-012  | Dubai      | 33        | 2026-09-01   |
| SKU-015  | Abu Dhabi  | 61        | 2026-09-29   |
| SKU-009  | Dubai      | 77        | 2026-10-15   |

Day index is 0-based from the window start (2026-07-10). Each of these
`(sku, location, date)` combinations is **completely missing** from the
file — the row does not exist. Participants must handle data gaps.

---

## 5. Aging Stock (older than 120 days)

**Location:** `inventory.csv` → `stock_age_days` column

| SKU      | Location   | Approx. Age Range |
|----------|------------|-------------------|
| SKU-010  | Sharjah    | 130–210 days      |
| SKU-014  | Sharjah    | 130–210 days      |
| SKU-018  | Dubai      | 130–190 days      |

All other inventory is aged 5–110 days. These three entries flag
slow-moving / obsolete stock that may need markdown or clearance.

---

## 6. Bad Transfer — cost exceeds benefit (Abu Dhabi → Dubai)

**Location:** `transfer_costs.csv`

The Abu Dhabi → Dubai transfer route has:
- `cost_per_unit` = **12.50** (vs. typical 0.50–3.00)
- `lead_time_days` = **5** (vs. typical 1–3)

**Why it's a trap:** For the hero SKU-001, Abu Dhabi is the
**short** location (stock 100, demand 360) while Dubai is the **surplus**
location (stock 500, demand 210). Transferring Abu Dhabi → Dubai moves
stock *away* from the shortage, and the inflated transfer cost makes the
loss even worse. A naive "just transfer to balance" algorithm that picks
the wrong direction will destroy value here.

The correct transfer direction is **Dubai → Abu Dhabi** (and optionally
Sharjah → Abu Dhabi).

---

## 7. Disruption A — Component Shortage (3 SKUs)

**Location:** `disruptions.csv` (DIS-001)

- **Type:** `component_shortage`
- **Affected SKUs:** SKU-002, SKU-003, SKU-004 (all USB-C chargers)
- **Severity:** high
- **Status:** ongoing

**Linked effect in `inbound_orders.csv`:** The purchase orders for
SKU-002, SKU-003, and SKU-004 are marked `status = delayed` with their
`expected_arrival` pushed out by an extra **21 days**. Participants
should connect the disruption record to the delayed inbound POs and
re-plan accordingly.

---

## 8. Disruption B — Shipping Route Delay (Jebel Ali)

**Location:** `disruptions.csv` (DIS-002)

- **Type:** `shipping_route_delay`
- **Affected location:** Dubai
- **Affected SKUs:** ALL (all inbound to Dubai port)
- **Severity:** medium
- **Status:** ongoing

This delays all containers routed through Jebel Ali by 5–10 days.
Solutions should factor this into inbound arrival estimates for
Dubai-destined orders.

---

## Summary Table

| # | Problem | File(s) | Key Signal |
|---|---------|---------|------------|
| 1 | Hero case: charger stock imbalance | inventory, sales | SKU-001 stock vs. demand |
| 2 | 10-day stockout (test period) | sales_history | SKU-007 Abu Dhabi, 0 units |
| 3 | 8-day stockout (training period) | sales_history | SKU-013 Dubai, 0 units |
| 4 | Missing sales records | sales_history | 4 absent rows |
| 5 | Aging stock (>120 days) | inventory | stock_age_days on 3 rows |
| 6 | Bad transfer (cost > benefit) | transfer_costs | Abu Dhabi→Dubai, $12.50/unit |
| 7 | Component shortage (3 SKUs) | disruptions, inbound_orders | DIS-001 + delayed POs |
| 8 | Shipping route delay | disruptions | DIS-002, Dubai inbound |
