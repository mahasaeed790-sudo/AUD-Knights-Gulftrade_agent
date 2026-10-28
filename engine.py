#!/usr/bin/env python3
"""
Gulftrade Agent — Transfer Engine
=================================

Step 4 of the hackathon pipeline.

This module brings together the smart forecast, current inventory, inbound
orders, disruptions, and transfer costs to:

  1. Flag imbalances (shortage / excess / ageing / distorted history) per
     SKU and location.
  2. Apply disruptions only to the SKUs or routes they explicitly affect,
     with functions to toggle each disruption on or off.
  3. Recommend inter-location transfers using a greedy algorithm with
     economics (benefit vs. cost), accounting for stockout timing — if the
     transfer arrives after the destination projects to stock out, the gap
     days count as unavoidable lost sales and the benefit is reduced
     accordingly. Rejected transfers are listed separately with the reason.
  4. Compare the "before" (smart forecast, no transfers) and "after"
     (approved transfers applied) state to quantify impact.  The baseline.py
     naive figures are shown in a separate reference block.
  5. Produce ageing_actions.csv (stock older than 120 days) and
     replenishment_needs.csv (remaining shortage after transfers).
  6. Hold everything in a State object so that editing a record or toggling
     a disruption and re-running gives updated results.

All recommendations start with status "pending approval" — nothing executes
automatically.  Forecasts are estimates, not guarantees.

Usage:
    python3 engine.py
"""

import csv
import os
import math
from collections import defaultdict
from datetime import datetime, timedelta

# Reuse the forecast engine from forecast.py — do not duplicate.
from forecast import (
    detect_stockout_days,
    detect_missing_days,
    compute_ewma,
    EWMA_ALPHA,
    STOCKOUT_RUN_THRESHOLD,
    TODAY,
    HISTORY_START,
    HISTORY_DAYS,
)

# ===========================================================================
# ASSUMPTIONS — change these to tune the engine
# ===========================================================================

# Forecast horizon (days) — must match forecast.py / baseline.py.
FORECAST_HORIZON_DAYS = 30

# "As of" date — same as forecast.py.
AS_OF_DATE = TODAY

# Inbound cutoff: orders arriving on or before this date count toward stock.
INBOUND_CUTOFF = AS_OF_DATE + timedelta(days=FORECAST_HORIZON_DAYS)

# --- Economics assumptions (labelled as assumptions) ---

# Unit margin is ASSUMED to be 25% of unit_cost.
# In a real system this would come from pricing data; here it is an assumption.
UNIT_MARGIN_RATE = 0.25

# Holding cost rate per unit per day as a fraction of unit_cost.
# Matches baseline.py's assumption (0.05%).
HOLDING_COST_RATE = 0.0005

# Days over which holding cost is applied to excess units.
HOLDING_COST_PERIOD_DAYS = 30

# --- Disruption delay assumptions ---
# The disruptions CSV does not have a numeric delay-days column.  We extract
# what we can from the description and fall back to these constants.

# DIS-001 (component_shortage): the generate_data.py script already delayed
# the affected POs by 21 days in the inbound_orders.csv.  We use the same
# value so a freshly toggled disruption produces consistent results.
COMPONENT_SHORTAGE_DELAY_DAYS = 21

# DIS-002 (shipping_route_delay): description says "5-10 days".  We use the
# midpoint (7 days) as a single deterministic value.
SHIPPING_ROUTE_DELAY_DAYS = 7

# --- Stockout detection ---
STOCKOUT_RUN_THRESHOLD_ENGINE = STOCKOUT_RUN_THRESHOLD  # reuse forecast.py's value

# --- Ageing threshold ---
# Stock older than this many days is flagged for review / promotion.
AGEING_THRESHOLD_DAYS = 120

# ===========================================================================
# Paths
# ===========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# ===========================================================================
# CSV loading helper
# ===========================================================================
def load_csv(filename, base_dir=DATA_DIR):
    """Read a CSV and return a list of dict rows."""
    path = os.path.join(base_dir, filename)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_date(s):
    """Parse a YYYY-MM-DD or YYYY-MM-DD HH:MM:SS string into a datetime."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {s!r}")


# ===========================================================================
# State object — holds all inputs so re-running gives updated results
# ===========================================================================
class State:
    """
    Holds all loaded data and the current disruption toggle state.

    Editing a record here (e.g. changing available stock) and re-running
    the engine functions gives updated flags, recommendations, and impact.
    """

    def __init__(self):
        # --- Raw CSV data ---
        self.skus = load_csv("skus.csv")
        self.inventory = load_csv("inventory.csv")
        self.inbound_orders = load_csv("inbound_orders.csv")
        self.transfer_costs = load_csv("transfer_costs.csv")
        self.disruptions = load_csv("disruptions.csv")
        self.sales = load_csv("sales_history.csv")

        # --- Derived lookups ---
        # unit_cost[sku_id] = float
        self.unit_cost = {r["sku_id"]: float(r["unit_cost"]) for r in self.skus}

        # inv_lookup[(sku_id, location)] = inventory row dict
        self.inv_lookup = {}
        for r in self.inventory:
            self.inv_lookup[(r["sku_id"], r["location"])] = dict(r)

        # transfer_lookup[(from, to)] = {cost_per_unit, lead_time_days}
        self.transfer_lookup = {}
        for r in self.transfer_costs:
            self.transfer_lookup[(r["from_location"], r["to_location"])] = {
                "cost_per_unit": float(r["cost_per_unit"]),
                "lead_time_days": int(r["lead_time_days"]),
            }

        # --- Disruption toggles: disruption_id → True/False ---
        # All start "on" (active), matching the CSV status "ongoing".
        self.disruption_toggles = {}
        for d in self.disruptions:
            self.disruption_toggles[d["disruption_id"]] = True

        # --- All SKU/location keys ---
        self.all_keys = sorted(self.inv_lookup.keys())

        # --- Locations ---
        self.locations = sorted(set(r["location"] for r in self.inventory))

    def toggle_disruption(self, disruption_id, active):
        """Switch a disruption on (True) or off (False)."""
        if disruption_id not in self.disruption_toggles:
            raise ValueError(f"Unknown disruption_id: {disruption_id}")
        self.disruption_toggles[disruption_id] = active

    def get_active_disruptions(self):
        """Return list of disruption dicts that are currently toggled on."""
        return [d for d in self.disruptions
                if self.disruption_toggles.get(d["disruption_id"], False)]

    def edit_inventory(self, sku_id, location, field, value):
        """Edit a single inventory field and update the lookup."""
        key = (sku_id, location)
        if key not in self.inv_lookup:
            raise ValueError(f"No inventory row for {key}")
        self.inv_lookup[key][field] = str(value)


# ===========================================================================
# Smart forecast (reuses forecast.py's EWMA + stockout correction)
# ===========================================================================
def build_smart_forecast(state):
    """
    For each (sku, location), produce a smart 30-day forecast using the
    same method as forecast.py: detect stockouts, fill with average, apply
    EWMA.  Also compute forecast_low (P10) and forecast_high (P90).

    Returns a dict: (sku_id, location) → {
        forecast_30d, forecast_low, forecast_high,
        data_quality_flag, note, days_adjusted
    }
    """
    # Build per (sku, location) date→units map from sales history.
    sales_map = defaultdict(dict)
    for row in state.sales:
        sales_map[(row["sku_id"], row["location"])][row["date"]] = int(row["units_sold"])

    # Build the full 90-day date index (reuse forecast.py constants).
    all_dates = []
    for i in range(HISTORY_DAYS):
        d = HISTORY_START + timedelta(days=i)
        all_dates.append(d.strftime("%Y-%m-%d"))

    results = {}

    for key in state.all_keys:
        sku_id, location = key
        date_units = sales_map.get(key, {})

        # Build the full 90-day series with None for missing dates.
        daily_series = []
        for d_str in all_dates:
            units = date_units.get(d_str)  # None if missing
            daily_series.append((d_str, units))

        # Detect stockout days (runs of 3+ consecutive zero/missing days).
        stockout_days = detect_stockout_days(daily_series)

        # Detect missing days (not already flagged as stockout).
        missing_days = detect_missing_days(daily_series)
        missing_days -= stockout_days

        # Compute the average of "normal" (non-excluded) selling days.
        normal_values = []
        for d_str, units in daily_series:
            if d_str in stockout_days or d_str in missing_days:
                continue
            normal_values.append(units)
        fill_value = sum(normal_values) / len(normal_values) if normal_values else 0.0

        days_adjusted = len(stockout_days) + len(missing_days)

        # Build cleaned series: real values for normal days, fill for excluded.
        cleaned = []
        for d_str, units in daily_series:
            if d_str in stockout_days or d_str in missing_days:
                cleaned.append(fill_value)
            else:
                cleaned.append(units)

        # EWMA over the cleaned series.
        daily_ewma = compute_ewma(cleaned, EWMA_ALPHA)
        forecast_30d = daily_ewma * FORECAST_HORIZON_DAYS

        # P10 / P90 from the standard deviation of normal daily sales.
        if len(normal_values) > 1:
            mean_normal = sum(normal_values) / len(normal_values)
            variance = sum((x - mean_normal) ** 2 for x in normal_values) / (len(normal_values) - 1)
            sigma_daily = math.sqrt(variance)
        else:
            sigma_daily = 0.0
        sigma_30 = sigma_daily * math.sqrt(FORECAST_HORIZON_DAYS)
        z_p10 = -1.2816
        z_p90 = 1.2816
        forecast_low = max(0, forecast_30d + z_p10 * sigma_30)
        forecast_high = forecast_30d + z_p90 * sigma_30

        # Data quality flag and note.
        notes = []
        if stockout_days:
            notes.append(f"{len(stockout_days)} stockout days filled with avg demand ({fill_value:.1f}/day)")
        if missing_days:
            notes.append(f"{len(missing_days)} missing days filled with avg demand ({fill_value:.1f}/day)")
        if stockout_days:
            flag = "stockout_adjusted"
        elif missing_days:
            flag = "missing_records"
        else:
            flag = "ok"
        note = "; ".join(notes) if notes else "No adjustments needed."

        results[key] = {
            "forecast_30d": forecast_30d,
            "forecast_low": forecast_low,
            "forecast_high": forecast_high,
            "data_quality_flag": flag,
            "note": note,
            "days_adjusted": days_adjusted,
        }

    return results


# ===========================================================================
# Inbound quantities (with disruption adjustments)
# ===========================================================================
def get_inbound_qty(state, sku_id, location):
    """
    Return total inbound units arriving within 30 days for a given
    (sku_id, destination), applying any active disruptions.

    - Component shortage: delays the SKU's inbound orders by
      COMPONENT_SHORTAGE_DELAY_DAYS.  If the delayed arrival falls beyond
      the 30-day cutoff, those units are excluded from effective stock.
    - Shipping route delay: adds SHIPPING_ROUTE_DELAY_DAYS to transfer
      lead times (handled separately in the transfer recommender).  It
      also delays inbound orders destined for the affected location.
    """
    total = 0
    for row in state.inbound_orders:
        if row["sku_id"] != sku_id or row["destination"] != location:
            continue

        arrival = parse_date(row["expected_arrival"])

        # Check if this (sku_id, destination) is affected by a disruption.
        for d in state.get_active_disruptions():
            affected_skus = d["affected_skus"].split(";")
            affects_this_sku = "ALL" in affected_skus or sku_id in affected_skus
            affected_locs = d["affected_locations"].split(";")
            affects_this_loc = location in affected_locs

            if d["type"] == "component_shortage":
                # Delay the SKU's inbound by the stated days.
                if affects_this_sku:
                    arrival = arrival + timedelta(days=COMPONENT_SHORTAGE_DELAY_DAYS)

            elif d["type"] == "shipping_route_delay":
                # Delay inbound to the affected location.
                if affects_this_loc:
                    arrival = arrival + timedelta(days=SHIPPING_ROUTE_DELAY_DAYS)

        # Only count if it arrives within the 30-day window.
        if arrival <= INBOUND_CUTOFF:
            total += int(row["quantity"])

    return total


def get_transfer_lead_time(state, from_loc, to_loc):
    """
    Return the transfer lead time in days between two locations,
    applying any active shipping-route delays on the affected route only.
    """
    base = state.transfer_lookup.get((from_loc, to_loc), {})
    lead_time = base.get("lead_time_days", 0)

    for d in state.get_active_disruptions():
        if d["type"] == "shipping_route_delay":
            affected_locs = d["affected_locations"].split(";")
            # Apply delay if either endpoint is in the affected locations.
            if from_loc in affected_locs or to_loc in affected_locs:
                lead_time += SHIPPING_ROUTE_DELAY_DAYS

    return lead_time


def get_transfer_cost(state, from_loc, to_loc):
    """Return the transfer cost per unit between two locations."""
    base = state.transfer_lookup.get((from_loc, to_loc), {})
    return base.get("cost_per_unit", 0.0)


# ===========================================================================
# 1. IMBALANCE FLAGS
# ===========================================================================
def compute_imbalance_flags(state, forecasts):
    """
    For each (sku, location), compute:
      - effective_stock = available - reserved + inbound_within_30d
      - shortage: effective_stock < forecast_30d + safety_stock
      - excess: effective_stock > forecast_high + safety_stock
      - ageing: stock_age_days > 120
      - distorted_history: carry the data_quality_flag from forecast

    Returns a list of dicts with all fields and a one-line reason.
    """
    results = []

    for key in state.all_keys:
        sku_id, location = key
        inv = state.inv_lookup[key]
        fc = forecasts.get(key, {})

        available = int(inv["available"])
        reserved = int(inv["reserved"])
        inbound = get_inbound_qty(state, sku_id, location)
        effective_stock = available - reserved + inbound

        forecast_30d = fc.get("forecast_30d", 0)
        forecast_high = fc.get("forecast_high", 0)
        safety_stock = int(inv["safety_stock"])
        stock_age = int(inv["stock_age_days"])
        dq_flag = fc.get("data_quality_flag", "ok")

        # --- Flag: shortage ---
        # If effective stock is below forecast + safety stock, we are short.
        is_shortage = effective_stock < (forecast_30d + safety_stock)
        shortage_units = max(0, int(forecast_30d + safety_stock - effective_stock))

        # --- Flag: excess ---
        # If effective stock exceeds the P90 forecast + safety stock, we
        # are overstocked.
        is_excess = effective_stock > (forecast_high + safety_stock)
        excess_units = max(0, int(effective_stock - forecast_high - safety_stock))

        # --- Flag: ageing ---
        is_ageing = stock_age > AGEING_THRESHOLD_DAYS

        # --- Flag: distorted_history ---
        is_distorted = dq_flag != "ok"

        # --- Build flags list and reason ---
        flags = []
        reasons = []
        if is_shortage:
            flags.append("shortage")
            reasons.append(f"short {shortage_units} units (stock {effective_stock} < "
                          f"forecast {forecast_30d:.0f} + safety {safety_stock})")
        if is_excess:
            flags.append("excess")
            reasons.append(f"excess {excess_units} units (stock {effective_stock} > "
                          f"P90 {forecast_high:.0f} + safety {safety_stock})")
        if is_ageing:
            flags.append("ageing")
            reasons.append(f"stock age {stock_age} days (>120)")
        if is_distorted:
            flags.append("distorted_history")
            reasons.append(f"history flag: {dq_flag}")

        flag_str = ";".join(flags) if flags else "ok"
        reason = ". ".join(reasons) if reasons else "No imbalance detected."

        results.append({
            "sku_id": sku_id,
            "location": location,
            "unit_cost": f"{state.unit_cost.get(sku_id, 0):.2f}",
            "forecast_30d": f"{forecast_30d:.1f}",
            "forecast_high": f"{forecast_high:.1f}",
            "available_stock": available,
            "reserved_stock": reserved,
            "inbound_within_30d": inbound,
            "effective_stock": effective_stock,
            "safety_stock": safety_stock,
            "stock_age_days": stock_age,
            "shortage_units": shortage_units,
            "excess_units": excess_units,
            "flags": flag_str,
            "data_quality_flag": dq_flag,
            "reason": reason,
        })

    return results


# ===========================================================================
# 3. TRANSFER RECOMMENDER (greedy)
# ===========================================================================
def recommend_transfers(state, forecasts, imbalance_flags):
    """
    Greedy transfer recommender.

    For each SKU:
      1. Identify shortage locations (destinations) and surplus locations
         (sources).
      2. Surplus available to move = effective_stock - forecast_30d -
         safety_stock (never below zero, never touching reserved stock).
      3. For each (source, destination) pair, compute the transfer
         quantity = min(destination shortage, source surplus).
      4. Check that the transfer lead time allows arrival within 30 days.
      5. Stockout timing: compute days_until_stockout at the destination =
         effective_stock / (forecast_30d / 30).  If the transfer lead time
         (including disruption delay) exceeds that, the gap days count as
         unavoidable lost sales: reduce the shortage avoided by
         gap_days × daily_demand, recompute benefit and net_benefit, and
         note "arrives about N days after projected stockout" in the reason.
         If net_benefit becomes <= 0, the transfer moves to not_recommended.
      6. Recommend only if net_benefit > 0.
      7. Rejected transfers go to a separate "not recommended" list with
         the reason.
      8. Rank recommendations by net_benefit (descending).

    All recommendations have status "pending approval".

    Returns: (recommendations, not_recommended, remaining_shortage)
    """
    recommendations = []
    not_recommended = []

    # Group imbalance flags by sku_id for easy access.
    flags_by_sku = defaultdict(dict)
    for f in imbalance_flags:
        flags_by_sku[f["sku_id"]][f["location"]] = f

    # Get all unique SKU IDs.
    all_sku_ids = sorted(set(f["sku_id"] for f in imbalance_flags))

    remaining_shortage_rows = []

    for sku_id in all_sku_ids:
        loc_flags = flags_by_sku[sku_id]
        cost = state.unit_cost.get(sku_id, 0)
        unit_margin = cost * UNIT_MARGIN_RATE  # ASSUMPTION: 25% of unit_cost

        # --- Identify shortage and surplus locations ---
        shortage_locs = []   # (location, shortage_units, effective_stock)
        surplus_locs = []    # (location, surplus_units, effective_stock)

        for loc, f in loc_flags.items():
            eff_stock = int(f["effective_stock"])
            fc = forecasts.get((sku_id, loc), {})
            forecast = fc.get("forecast_30d", 0)
            safety = int(f["safety_stock"])

            # Shortage: effective_stock < forecast + safety
            shortage_qty = max(0, int(forecast + safety - eff_stock))
            if shortage_qty > 0:
                shortage_locs.append((loc, shortage_qty, eff_stock))

            # Surplus: effective_stock - forecast - safety (never below zero,
            # never touching reserved stock).
            surplus_qty = max(0, eff_stock - int(forecast) - safety)
            if surplus_qty > 0:
                surplus_locs.append((loc, surplus_qty, eff_stock))

        # --- Match sources to destinations (greedy) ---
        # We'll track remaining surplus as we allocate.
        surplus_remaining = {loc: qty for loc, qty, _ in surplus_locs}
        shortage_remaining = {loc: qty for loc, qty, _ in shortage_locs}

        for dest_loc, dest_shortage, _ in shortage_locs:
            for src_loc, src_surplus, _ in surplus_locs:
                if src_loc == dest_loc:
                    continue

                # Quantity to transfer = min(destination shortage, source surplus).
                qty = min(shortage_remaining[dest_loc], surplus_remaining[src_loc])
                if qty <= 0:
                    continue

                # --- Lead time (includes any active disruption delay) ---
                lead_time = get_transfer_lead_time(state, src_loc, dest_loc)
                transfer_cost_per_unit = get_transfer_cost(state, src_loc, dest_loc)

                # The transfer must arrive within the 30-day window.
                arrival_day = lead_time  # days from now
                arrives_in_time = arrival_day <= FORECAST_HORIZON_DAYS

                # --- Stockout timing at the destination ---
                # Days until the destination runs out of stock, assuming even
                # demand: effective_stock / (forecast_30d / 30).
                # If daily demand is zero, there is no stockout risk.
                dest_eff = int(loc_flags[dest_loc]["effective_stock"])
                fc_dest = forecasts.get((sku_id, dest_loc), {})
                forecast_dest = fc_dest.get("forecast_30d", 0)
                daily_demand = forecast_dest / FORECAST_HORIZON_DAYS
                if daily_demand > 0:
                    days_until_stockout = dest_eff / daily_demand
                else:
                    days_until_stockout = float("inf")

                # If the transfer lead time exceeds days_until_stockout, the
                # destination will be out of stock for gap_days before the
                # shipment arrives.  Those gap days are unavoidable lost sales:
                # reduce the shortage avoided by gap_days × daily_demand.
                gap_days = 0
                lost_sales_units = 0
                if days_until_stockout != float("inf") and lead_time > days_until_stockout:
                    gap_days = lead_time - int(days_until_stockout)
                    lost_sales_units = gap_days * daily_demand
                    # The transfer can only avoid shortage for the units that
                    # would still be demanded after the shipment arrives.
                    # Cap lost sales at the transfer quantity (can't lose more
                    # sales than the units you were going to send).
                    lost_sales_units = min(lost_sales_units, qty)

                # --- Economics ---
                # Effective shortage avoided = transfer quantity minus
                # unavoidable lost-sales units (stockout gap before arrival).
                shortage_avoided = qty - lost_sales_units

                # Benefit: shortage avoided × unit_margin
                #          + holding cost saved at the source for the moved units
                benefit_shortage_avoided = shortage_avoided * unit_margin
                holding_saved = qty * cost * HOLDING_COST_RATE * HOLDING_COST_PERIOD_DAYS
                benefit = benefit_shortage_avoided + holding_saved

                # Cost: quantity × transfer cost per unit
                transfer_cost = qty * transfer_cost_per_unit

                net_benefit = benefit - transfer_cost

                # --- Confidence note ---
                if fc_dest.get("data_quality_flag", "ok") != "ok":
                    confidence = "LOW — destination demand history is distorted"
                else:
                    confidence = "MEDIUM — based on 90-day EWMA forecast"

                # --- Plain-English reason ---
                reason_parts = [
                    f"Move {qty} units of {sku_id} from {src_loc} to {dest_loc}: "
                    f"shortage of {dest_shortage} units at destination, "
                    f"surplus of {src_surplus} units at source.",
                    f"Transfer cost ${transfer_cost_per_unit:.2f}/unit × {qty} = ${transfer_cost:.2f}.",
                    f"Benefit: {shortage_avoided:.0f} × ${unit_margin:.2f} margin + ${holding_saved:.2f} holding saved = ${benefit:.2f}.",
                    f"Net benefit ${net_benefit:.2f}.",
                    f"Lead time {lead_time} days.",
                ]
                if gap_days > 0:
                    reason_parts.append(
                        f"Arrives about {gap_days} days after projected stockout "
                        f"(stockout in {days_until_stockout:.0f} days, lead time {lead_time} days); "
                        f"{lost_sales_units:.0f} units treated as unavoidable lost sales."
                    )
                reason = " ".join(reason_parts)

                if not arrives_in_time:
                    not_recommended.append({
                        "sku_id": sku_id,
                        "from_location": src_loc,
                        "to_location": dest_loc,
                        "quantity": qty,
                        "transfer_cost_per_unit": f"{transfer_cost_per_unit:.2f}",
                        "transfer_cost_total": f"{transfer_cost:.2f}",
                        "lead_time_days": lead_time,
                        "net_benefit": f"{net_benefit:.2f}",
                        "reason": reason + " REJECTED: transfer lead time exceeds 30-day window.",
                    })
                    continue

                if net_benefit > 0:
                    # Recommend!
                    recommendations.append({
                        "sku_id": sku_id,
                        "from_location": src_loc,
                        "to_location": dest_loc,
                        "quantity": qty,
                        "transfer_cost_per_unit": f"{transfer_cost_per_unit:.2f}",
                        "transfer_cost_total": f"{transfer_cost:.2f}",
                        "lead_time_days": lead_time,
                        "net_benefit": f"{net_benefit:.2f}",
                        "benefit": f"{benefit:.2f}",
                        "unit_margin": f"{unit_margin:.2f}",
                        "holding_saved": f"{holding_saved:.2f}",
                        "days_until_stockout": f"{days_until_stockout:.0f}" if days_until_stockout != float("inf") else "—",
                        "lost_sales_units": f"{lost_sales_units:.0f}",
                        "confidence": confidence,
                        "status": "pending approval",
                        "reason": reason,
                    })

                    # Reduce remaining shortage and surplus.
                    shortage_remaining[dest_loc] -= qty
                    surplus_remaining[src_loc] -= qty
                else:
                    # Rejected: cost exceeds (adjusted) benefit.
                    reject_reason = (
                        f" REJECTED: transfer cost exceeds benefit (net ${net_benefit:.2f})"
                    )
                    if gap_days > 0:
                        reject_reason += (
                            f", partly because {lost_sales_units:.0f} units are lost "
                            f"to the {gap_days}-day stockout gap before arrival"
                        )
                    reject_reason += "."
                    not_recommended.append({
                        "sku_id": sku_id,
                        "from_location": src_loc,
                        "to_location": dest_loc,
                        "quantity": qty,
                        "transfer_cost_per_unit": f"{transfer_cost_per_unit:.2f}",
                        "transfer_cost_total": f"{transfer_cost:.2f}",
                        "lead_time_days": lead_time,
                        "net_benefit": f"{net_benefit:.2f}",
                        "reason": reason + reject_reason,
                    })

        # --- Remaining shortage that transfers cannot fix ---
        for dest_loc, _, _ in shortage_locs:
            remaining = shortage_remaining[dest_loc]
            if remaining > 0:
                remaining_shortage_rows.append({
                    "sku_id": sku_id,
                    "location": dest_loc,
                    "remaining_shortage_units": remaining,
                    "label": "needs replenishment",
                    "reason": f"After all viable transfers, {remaining} units of shortage "
                              f"remain at {dest_loc} for {sku_id}. Place a replenishment order.",
                })

    # Rank recommendations by net_benefit (descending).
    recommendations.sort(key=lambda r: float(r["net_benefit"]), reverse=True)

    return recommendations, not_recommended, remaining_shortage_rows


# ===========================================================================
# 4. IMPACT COMPARISON
# ===========================================================================
def compute_impact(state, forecasts, recommendations, imbalance_flags):
    """
    Compare three states:
      1. baseline_py    — totals from baseline_results.csv (naive forecast,
                          separate reference block, not directly comparable).
      2. before_transfers — smart forecast, NO transfers applied.
      3. after_transfers  — smart forecast WITH recommended transfers applied.

    The primary comparison is before_transfers vs after_transfers (same
    forecast method, isolating the effect of transfers). The baseline_py
    figures are in a separate "Naive baseline reference" block.

    Returns:
        (impact_rows, baseline_ref_rows, net_benefit_dollars)
    """
    # --- 1. baseline.py totals (naive forecast, separate reference block) ---
    baseline_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
    baseline_totals = {"shortage": 0, "excess": 0, "holding_cost": 0.0}
    if os.path.exists(baseline_path):
        for row in csv.DictReader(open(baseline_path, newline="")):
            baseline_totals["shortage"] += int(row["expected_shortage_units"])
            baseline_totals["excess"] += int(row["expected_excess_units"])
            baseline_totals["holding_cost"] += float(row["holding_cost_30d"])
    else:
        print("WARNING: baseline_results.csv not found — baseline_py metrics will be zero.")

    # Baseline reference rows (separate block, not directly comparable).
    baseline_ref_rows = [
        {"metric": "shortage_units", "value": baseline_totals["shortage"]},
        {"metric": "excess_units", "value": baseline_totals["excess"]},
        {"metric": "holding_cost", "value": f"{baseline_totals['holding_cost']:.2f}"},
    ]

    # --- Transfer adjustments ---
    # transfers move qty from source to destination.
    transfer_adjustments = defaultdict(int)  # (sku, loc) → delta
    total_transfer_cost = 0.0
    for rec in recommendations:
        qty = int(rec["quantity"])
        transfer_adjustments[(rec["sku_id"], rec["to_location"])] += qty
        transfer_adjustments[(rec["sku_id"], rec["from_location"])] -= qty
        total_transfer_cost += float(rec["transfer_cost_total"])

    # --- 2 & 3. Before and after transfers (same smart forecast) ---
    # Before = effective stock as-is. After = effective stock + transfer delta.
    def compute_totals(apply_transfers):
        """Compute shortage, excess, holding cost, ageing units."""
        totals = {"shortage": 0, "excess": 0, "holding_cost": 0.0, "ageing_units": 0}
        for f in imbalance_flags:
            sku_id = f["sku_id"]
            loc = f["location"]
            adj = transfer_adjustments.get((sku_id, loc), 0) if apply_transfers else 0
            eff_stock = int(f["effective_stock"]) + adj
            forecast = float(f["forecast_30d"])
            forecast_high = float(f["forecast_high"])
            safety = int(f["safety_stock"])
            cost = state.unit_cost.get(sku_id, 0)

            # Shortage: effective stock below forecast + safety.
            shortage = max(0, int(forecast + safety - eff_stock))
            # Excess: effective stock above P90 forecast + safety.
            excess = max(0, int(eff_stock - forecast_high - safety))

            totals["shortage"] += shortage
            totals["excess"] += excess
            totals["holding_cost"] += excess * cost * HOLDING_COST_RATE * HOLDING_COST_PERIOD_DAYS

            # Ageing units: stock older than threshold (doesn't change with transfers).
            if int(f["stock_age_days"]) > AGEING_THRESHOLD_DAYS:
                totals["ageing_units"] += int(f["effective_stock"])
        return totals

    before = compute_totals(apply_transfers=False)
    after = compute_totals(apply_transfers=True)

    # --- Net benefit in dollar terms ---
    # shortage_reduction × avg unit margin + holding cost saved − transfer cost
    avg_cost = sum(state.unit_cost.values()) / len(state.unit_cost) if state.unit_cost else 0
    avg_margin = avg_cost * UNIT_MARGIN_RATE
    shortage_reduction = before["shortage"] - after["shortage"]
    holding_saved = before["holding_cost"] - after["holding_cost"]
    net_benefit_dollars = (shortage_reduction * avg_margin) + holding_saved - total_transfer_cost

    # Primary comparison rows: "Before (smart forecast, no transfers)" vs
    # "After (approved transfers applied)".
    rows = [
        {"metric": "shortage_units",
         "before": before["shortage"], "after": after["shortage"],
         "delta": after["shortage"] - before["shortage"]},
        {"metric": "excess_units",
         "before": before["excess"], "after": after["excess"],
         "delta": after["excess"] - before["excess"]},
        {"metric": "ageing_units",
         "before": before["ageing_units"], "after": after["ageing_units"],
         "delta": after["ageing_units"] - before["ageing_units"]},
        {"metric": "holding_cost",
         "before": f"{before['holding_cost']:.2f}",
         "after": f"{after['holding_cost']:.2f}",
         "delta": f"{after['holding_cost'] - before['holding_cost']:.2f}"},
        {"metric": "transfer_cost",
         "before": "0.00",
         "after": f"{total_transfer_cost:.2f}",
         "delta": f"{total_transfer_cost:.2f}"},
        {"metric": "net_benefit",
         "before": "—",
         "after": f"{net_benefit_dollars:.2f}", "delta": "—"},
    ]

    return rows, baseline_ref_rows, net_benefit_dollars


# ===========================================================================
# Ageing stock actions
# ===========================================================================
def build_ageing_actions(state, recommendations):
    """
    Build ageing action rows for every inventory line with stock_age_days >
    AGEING_THRESHOLD_DAYS (120).

    Columns: sku, location, units, age, action.

    Action text: "planner review: prioritise selling, or consider promotion".
    If a transfer is recommended FROM that SKU and location, append
    "ship oldest stock first".
    """
    # Build a set of (sku_id, from_location) pairs that have an approved
    # recommendation shipping out of them.
    recommended_sources = set()
    for rec in recommendations:
        recommended_sources.add((rec["sku_id"], rec["from_location"]))

    actions = []
    for key in state.all_keys:
        sku_id, location = key
        inv = state.inv_lookup[key]
        stock_age = int(inv["stock_age_days"])
        if stock_age > AGEING_THRESHOLD_DAYS:
            available = int(inv["available"])
            action = "planner review: prioritise selling, or consider promotion"
            if (sku_id, location) in recommended_sources:
                action += "; ship oldest stock first"
            actions.append({
                "sku": sku_id,
                "location": location,
                "units": available,
                "age": stock_age,
                "action": action,
            })

    # Sort by age descending (oldest first).
    actions.sort(key=lambda r: r["age"], reverse=True)
    return actions


# ===========================================================================
# Replenishment needs (remaining shortage after transfers)
# ===========================================================================
def build_replenishment_needs(state, remaining_shortage):
    """
    Convert remaining shortage rows into replenishment_needs.csv rows with:
      sku, location, units, supplier_lead_time_days, delayed_by_disruption.

    The supplier lead time is looked up from the most recent inbound order
    for that (sku, destination).  delayed_by_disruption is "yes" if any
    active disruption affects that SKU or destination location.
    """
    needs = []
    for r in remaining_shortage:
        sku_id = r["sku_id"]
        location = r["location"]

        # Find the supplier lead time from the inbound orders for this SKU.
        # Use the first matching order's supplier_lead_time_days; if none,
        # default to 14 days (a reasonable planning assumption).
        supplier_lead_time = 14  # ASSUMPTION: default if no PO found
        for row in state.inbound_orders:
            if row["sku_id"] == sku_id and row["destination"] == location:
                supplier_lead_time = int(row["supplier_lead_time_days"])
                break

        # Check if any active disruption affects this SKU or location.
        delayed = "no"
        for d in state.get_active_disruptions():
            affected_skus = d["affected_skus"].split(";")
            affects_sku = "ALL" in affected_skus or sku_id in affected_skus
            affected_locs = d["affected_locations"].split(";")
            affects_loc = location in affected_locs
            if affects_sku or affects_loc:
                delayed = "yes"
                break

        needs.append({
            "sku": sku_id,
            "location": location,
            "units": r["remaining_shortage_units"],
            "supplier_lead_time_days": supplier_lead_time,
            "delayed_by_disruption": delayed,
        })

    # Sort by units descending.
    needs.sort(key=lambda r: r["units"], reverse=True)
    return needs


# ===========================================================================
# CSV writer
# ===========================================================================
def write_csv(filename, fieldnames, rows):
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ===========================================================================
# Run the full pipeline and save outputs
# ===========================================================================
def run_engine(state):
    """
    Run the full engine pipeline on the given State object.

    Returns a dict with all results.
    """
    # 1. Smart forecast.
    forecasts = build_smart_forecast(state)

    # 2. Imbalance flags.
    imbalance_flags = compute_imbalance_flags(state, forecasts)

    # 3. Transfer recommendations.
    recommendations, not_recommended, remaining_shortage = recommend_transfers(
        state, forecasts, imbalance_flags
    )

    # 4. Impact comparison.
    impact_rows, baseline_ref_rows, net_benefit = compute_impact(
        state, forecasts, recommendations, imbalance_flags
    )

    # 5. Ageing actions (stock older than threshold).
    ageing_actions = build_ageing_actions(state, recommendations)

    # 6. Replenishment needs (remaining shortage after transfers).
    replenishment_needs = build_replenishment_needs(state, remaining_shortage)

    return {
        "forecasts": forecasts,
        "imbalance_flags": imbalance_flags,
        "recommendations": recommendations,
        "not_recommended": not_recommended,
        "remaining_shortage": remaining_shortage,
        "impact_rows": impact_rows,
        "baseline_ref_rows": baseline_ref_rows,
        "net_benefit": net_benefit,
        "ageing_actions": ageing_actions,
        "replenishment_needs": replenishment_needs,
    }


def save_outputs(results):
    """Save all result sets to /output CSV files."""
    # imbalance_flags.csv
    write_csv("imbalance_flags.csv", [
        "sku_id", "location", "unit_cost", "forecast_30d", "forecast_high",
        "available_stock", "reserved_stock", "inbound_within_30d",
        "effective_stock", "safety_stock", "stock_age_days",
        "shortage_units", "excess_units", "flags", "data_quality_flag", "reason",
    ], results["imbalance_flags"])

    # recommendations.csv
    write_csv("recommendations.csv", [
        "sku_id", "from_location", "to_location", "quantity",
        "transfer_cost_per_unit", "transfer_cost_total", "lead_time_days",
        "net_benefit", "benefit", "unit_margin", "holding_saved",
        "days_until_stockout", "lost_sales_units",
        "confidence", "status", "reason",
    ], results["recommendations"])

    # not_recommended.csv
    write_csv("not_recommended.csv", [
        "sku_id", "from_location", "to_location", "quantity",
        "transfer_cost_per_unit", "transfer_cost_total", "lead_time_days",
        "net_benefit", "reason",
    ], results["not_recommended"])

    # impact_summary.csv (primary: before vs after transfers)
    write_csv("impact_summary.csv", [
        "metric", "before", "after", "delta",
    ], results["impact_rows"])

    # baseline_reference.csv (separate block — not directly comparable)
    write_csv("baseline_reference.csv", [
        "metric", "value",
    ], results["baseline_ref_rows"])

    # ageing_actions.csv
    write_csv("ageing_actions.csv", [
        "sku", "location", "units", "age", "action",
    ], results["ageing_actions"])

    # replenishment_needs.csv
    write_csv("replenishment_needs.csv", [
        "sku", "location", "units", "supplier_lead_time_days",
        "delayed_by_disruption",
    ], results["replenishment_needs"])


def print_results(results, label=""):
    """Print a human-readable summary of the engine results."""
    if label:
        print(f"\n{'=' * 60}")
        print(f" {label}")
        print(f"{'=' * 60}")

    # --- Hero case recommendation ---
    print("\n--- SKU-001 Hero Case Recommendations ---")
    hero_recs = [r for r in results["recommendations"] if r["sku_id"] == "SKU-001"]
    if hero_recs:
        for r in hero_recs:
            print(f"  {r['from_location']} → {r['to_location']}: "
                  f"{r['quantity']} units, net benefit ${r['net_benefit']}, "
                  f"lead time {r['lead_time_days']}d, status: {r['status']}")
            print(f"    Reason: {r['reason']}")
    else:
        print("  No recommendations for SKU-001.")

    # --- Not recommended ---
    print("\n--- Not Recommended Transfers ---")
    if results["not_recommended"]:
        for r in results["not_recommended"]:
            print(f"  {r['sku_id']}: {r['from_location']} → {r['to_location']}: "
                  f"{r['quantity']} units, net ${r['net_benefit']}")
            reason = r["reason"]
            rejected_part = reason[reason.index("REJECTED"):] if "REJECTED" in reason else reason
            print(f"    {rejected_part}")
    else:
        print("  (none)")

    # --- Impact summary (before vs after transfers) ---
    print("\n--- Impact Summary ---")
    print("  Before (smart forecast, no transfers) vs After (approved transfers applied)")
    print(f"  {'Metric':<20} {'Before':>12} {'After':>12} {'Delta':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    for row in results["impact_rows"]:
        print(f"  {row['metric']:<20} {str(row['before']):>12} {str(row['after']):>12} {str(row['delta']):>12}")
    print(f"\n  Net benefit (estimated): ${results['net_benefit']:.2f}")
    print("  NOTE: Forecasts are estimates, not guarantees. "
          "All transfers are 'pending approval'.")

    # --- Baseline reference (separate block) ---
    print("\n--- Naive baseline reference (14-day average, no safety stock; "
          "not directly comparable) ---")
    for row in results["baseline_ref_rows"]:
        print(f"  {row['metric']:<20} {str(row['value']):>12}")

    # --- Ageing actions ---
    print("\n--- Ageing Stock Actions (stock_age > 120 days) ---")
    if results["ageing_actions"]:
        print(f"  {'SKU':<10} {'Location':<12} {'Units':>6} {'Age':>5}  Action")
        print(f"  {'-'*10} {'-'*12} {'-'*6} {'-'*5}  {'-'*40}")
        for r in results["ageing_actions"]:
            print(f"  {r['sku']:<10} {r['location']:<12} {r['units']:>6} {r['age']:>5}  {r['action']}")
    else:
        print("  (none)")

    # --- Replenishment needs (replaces long remaining-shortage printout) ---
    print("\n--- Replenishment Needs ---")
    needs = results["replenishment_needs"]
    total_units = sum(r["units"] for r in needs)
    print(f"  Total remaining shortage (needs replenishment): {total_units:,} units "
          f"across {len(needs)} SKU/location lines.")
    if needs:
        print(f"  Top 10 by units:")
        print(f"  {'SKU':<10} {'Location':<12} {'Units':>6} {'Lead':>5}  Delayed")
        print(f"  {'-'*10} {'-'*12} {'-'*6} {'-'*5}  {'-'*8}")
        for r in needs[:10]:
            print(f"  {r['sku']:<10} {r['location']:<12} {r['units']:>6} "
                  f"{r['supplier_lead_time_days']:>5}  {r['delayed_by_disruption']}")
    else:
        print("  (none — all shortage covered by transfers)")


# ===========================================================================
# Disruption scope printing
# ===========================================================================
def print_disruption_scope(state):
    """
    For each disruption (whether on or off), print:
      - Which SKUs it affects
      - Which locations / routes it affects
      - Which inventory or inbound rows change when it is toggled on
    """
    print("\n--- Disruption Scope ---")
    for d in state.disruptions:
        did = d["disruption_id"]
        is_on = state.disruption_toggles.get(did, False)
        status_str = "ON" if is_on else "OFF"
        affected_skus = d["affected_skus"]
        affected_locs = d["affected_locations"]
        dtype = d["type"]
        desc = d["description"]

        print(f"\n  {did} ({dtype}) — currently {status_str}")
        print(f"    Description : {desc}")
        if affected_skus == "ALL":
            print(f"    Affected SKUs: ALL")
        else:
            print(f"    Affected SKUs: {affected_skus}")

        if dtype == "component_shortage":
            # Affects inbound orders for the affected SKUs (delayed by
            # COMPONENT_SHORTAGE_DELAY_DAYS, possibly pushing beyond the
            # 30-day window).
            print(f"    Affected locations: {affected_locs}")
            delay = COMPONENT_SHORTAGE_DELAY_DAYS
            sku_list = affected_skus.split(";")
            changed = []
            for row in state.inbound_orders:
                if row["sku_id"] not in sku_list:
                    continue
                arrival = parse_date(row["expected_arrival"])
                delayed_arrival = arrival + timedelta(days=delay)
                was_in = arrival <= INBOUND_CUTOFF
                now_in = delayed_arrival <= INBOUND_CUTOFF
                if was_in or now_in:
                    changed.append({
                        "order_id": row["order_id"],
                        "sku": row["sku_id"],
                        "destination": row["destination"],
                        "original_arrival": arrival.strftime("%Y-%m-%d"),
                        "delayed_arrival": delayed_arrival.strftime("%Y-%m-%d"),
                        "within_30d_before": "yes" if was_in else "no",
                        "within_30d_after": "yes" if now_in else "no",
                    })
            if changed:
                print(f"    Delay applied: +{delay} days to inbound POs")
                print(f"    Rows changed when ON ({len(changed)} POs):")
                for c in changed:
                    drop_note = ""
                    if c["within_30d_before"] == "yes" and c["within_30d_after"] == "no":
                        drop_note = " ⚠ drops out of 30-day window"
                    print(f"      {c['order_id']} ({c['sku']} → {c['destination']}): "
                          f"{c['original_arrival']} → {c['delayed_arrival']} "
                          f"(in-30d: {c['within_30d_before']}→{c['within_30d_after']}){drop_note}")
            else:
                print(f"    Delay applied: +{delay} days (no matching inbound POs)")

        elif dtype == "shipping_route_delay":
            # Affects transfers to/from the affected locations, and inbound
            # orders destined for those locations.
            loc_list = affected_locs.split(";")
            print(f"    Affected locations: {affected_locs}")
            delay = SHIPPING_ROUTE_DELAY_DAYS

            # Show affected transfer routes.
            print(f"    Affected transfer routes (lead time +{delay} days):")
            for r in state.transfer_costs:
                frm = r["from_location"]
                to = r["to_location"]
                if frm in loc_list or to in loc_list:
                    base_lt = int(r["lead_time_days"])
                    delayed_lt = base_lt + delay
                    print(f"      {frm} → {to}: {base_lt}d → {delayed_lt}d")

            # Show affected inbound orders.
            changed = []
            for row in state.inbound_orders:
                if row["destination"] not in loc_list:
                    continue
                arrival = parse_date(row["expected_arrival"])
                delayed_arrival = arrival + timedelta(days=delay)
                was_in = arrival <= INBOUND_CUTOFF
                now_in = delayed_arrival <= INBOUND_CUTOFF
                if was_in or now_in:
                    changed.append({
                        "order_id": row["order_id"],
                        "sku": row["sku_id"],
                        "destination": row["destination"],
                        "original_arrival": arrival.strftime("%Y-%m-%d"),
                        "delayed_arrival": delayed_arrival.strftime("%Y-%m-%d"),
                        "within_30d_before": "yes" if was_in else "no",
                        "within_30d_after": "yes" if now_in else "no",
                    })
            if changed:
                print(f"    Inbound POs delayed when ON ({len(changed)} POs):")
                for c in changed:
                    drop_note = ""
                    if c["within_30d_before"] == "yes" and c["within_30d_after"] == "no":
                        drop_note = " ⚠ drops out of 30-day window"
                    print(f"      {c['order_id']} ({c['sku']} → {c['destination']}): "
                          f"{c['original_arrival']} → {c['delayed_arrival']} "
                          f"(in-30d: {c['within_30d_before']}→{c['within_30d_after']}){drop_note}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Gulftrade Transfer Engine")
    print(f"  As-of date        : {AS_OF_DATE.date()}")
    print(f"  Forecast horizon  : {FORECAST_HORIZON_DAYS} days")
    print(f"  Unit margin rate  : {UNIT_MARGIN_RATE:.0%} of unit_cost (ASSUMPTION)")
    print(f"  Holding cost rate : {HOLDING_COST_RATE:.4%} of unit_cost / unit / day")
    print(f"  Ageing threshold  : {AGEING_THRESHOLD_DAYS} days")
    print(f"  Component shortage delay : {COMPONENT_SHORTAGE_DELAY_DAYS} days")
    print(f"  Shipping route delay     : {SHIPPING_ROUTE_DELAY_DAYS} days\n")

    # ------------------------------------------------------------------
    # Print disruption scope (which SKUs / routes each disruption affects)
    # ------------------------------------------------------------------
    state_scope = State()
    print_disruption_scope(state_scope)

    # ------------------------------------------------------------------
    # Run with ALL disruptions ON (default state) and save outputs
    # ------------------------------------------------------------------
    state_all = State()
    results_all = run_engine(state_all)
    save_outputs(results_all)
    print("\nOutputs saved to /output:")
    print("  imbalance_flags.csv, recommendations.csv, not_recommended.csv,")
    print("  impact_summary.csv, baseline_reference.csv,")
    print("  ageing_actions.csv, replenishment_needs.csv")

    print_results(results_all, label="ALL DISRUPTIONS ON")

    # ------------------------------------------------------------------
    # SKU-001 hero case: disruptions OFF vs shipping delay (DIS-002) ON
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(" SKU-001 HERO CASE: Disruptions OFF vs Shipping Delay ON")
    print(f"{'=' * 60}")

    # --- Disruptions OFF ---
    state_off = State()
    for did in state_off.disruption_toggles:
        state_off.toggle_disruption(did, False)
    results_off = run_engine(state_off)

    print("\n  ▸ Disruptions OFF (baseline lead time Dubai→Abu Dhabi = 3 days):")
    hero_off = [r for r in results_off["recommendations"] if r["sku_id"] == "SKU-001"]
    if hero_off:
        for r in hero_off:
            print(f"    {r['from_location']} → {r['to_location']}: {r['quantity']} units, "
                  f"net ${r['net_benefit']}, lead {r['lead_time_days']}d")
            print(f"    Reason: {r['reason']}")
    else:
        print("    No recommendations for SKU-001.")

    # --- Only DIS-002 (shipping route delay) ON ---
    state_002 = State()
    state_002.toggle_disruption("DIS-001", False)
    results_002 = run_engine(state_002)

    print("\n  ▸ DIS-002 ON (shipping delay +7 days, Dubai→Abu Dhabi = 10 days):")
    hero_002 = [r for r in results_002["recommendations"] if r["sku_id"] == "SKU-001"]
    if hero_002:
        for r in hero_002:
            print(f"    {r['from_location']} → {r['to_location']}: {r['quantity']} units, "
                  f"net ${r['net_benefit']}, lead {r['lead_time_days']}d")
            print(f"    Reason: {r['reason']}")
    else:
        # Check if it was rejected
        hero_rej = [r for r in results_002["not_recommended"] if r["sku_id"] == "SKU-001"]
        if hero_rej:
            for r in hero_rej:
                print(f"    {r['from_location']} → {r['to_location']}: {r['quantity']} units, "
                      f"net ${r['net_benefit']}, lead {r['lead_time_days']}d")
                print(f"    Reason: {r['reason']}")
        else:
            print("    No recommendations or rejections for SKU-001.")

    # --- Side-by-side summary ---
    print("\n  ▸ Comparison:")
    off_net = float(hero_off[0]["net_benefit"]) if hero_off else 0
    on_net = float(hero_002[0]["net_benefit"]) if hero_002 else 0
    print(f"    Disruptions OFF : net benefit ${off_net:.2f}, lead time "
          f"{hero_off[0]['lead_time_days'] if hero_off else '—'}d")
    print(f"    DIS-002 ON      : net benefit ${on_net:.2f}, lead time "
          f"{hero_002[0]['lead_time_days'] if hero_002 else '—'}d")

    # ------------------------------------------------------------------
    # Full results: ALL disruptions OFF
    # ------------------------------------------------------------------
    print_results(results_off, label="ALL DISRUPTIONS OFF")

    # ------------------------------------------------------------------
    # Full results: ONLY DIS-001 (component shortage) ON
    # ------------------------------------------------------------------
    state_001 = State()
    state_001.toggle_disruption("DIS-002", False)
    results_001 = run_engine(state_001)
    print_results(results_001, label="DIS-001 ONLY (component shortage)")

    # ------------------------------------------------------------------
    # Full results: ONLY DIS-002 (shipping route delay) ON
    # ------------------------------------------------------------------
    print_results(results_002, label="DIS-002 ONLY (shipping route delay)")


if __name__ == "__main__":
    main()
