#!/usr/bin/env python3
"""
Gulftrade Agent — Baseline Demand Forecast & Stock Analysis
============================================================

Step 2 of the hackathon pipeline.

This script loads the generated CSVs in ./data/, builds a *naive* 30-day
demand forecast per SKU per location, compares it to current effective
stock (on-hand + inbound within 30 days), and reports shortages, excess,
and an estimated holding cost.

Assumptions (the "baseline" — no transfers, no reordering):
  1. Forecast = average daily sales over the last 14 days × 30.
     Zero-sales days (e.g. stockouts) are included in the average, so a
     stockout artificially lowers the forecast — this is the naive baseline.
  2. Effective stock = (available − reserved) + inbound arriving within
     the next 30 days.
  3. Holding cost = 0.05 % of unit_cost per unit per day, applied to
     excess units for 30 days.
  4. No stock transfers between locations (baseline only).

Usage:
    python3 baseline.py
"""

import csv
import os
from collections import defaultdict
from datetime import datetime, timedelta

# ===========================================================================
# ASSUMPTIONS — change these to tune the baseline model
# ===========================================================================

# How many recent days of sales to average for the forecast.
LOOKBACK_DAYS = 14

# Forecast horizon in days.
FORECAST_HORIZON_DAYS = 30

# Today's date — the "as of" date for the analysis.  We derive it from the
# latest sales date in the data so the script is always aligned with the
# dataset.  (The data window ends 2026-10-27.)
TODAY = datetime(2026, 10, 27)

# Cutoff for inbound orders that count toward effective stock: anything
# arriving on or before this date is included.
INBOUND_CUTOFF = TODAY + timedelta(days=FORECAST_HORIZON_DAYS)

# Holding cost rate per unit per day, as a fraction of unit_cost.
# 0.05 %  →  0.0005
HOLDING_COST_RATE = 0.0005

# Number of days over which holding cost is applied to excess units.
HOLDING_COST_PERIOD_DAYS = 30

# ===========================================================================
# Paths
# ===========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# ===========================================================================
# CSV loading helpers
# ===========================================================================
def load_csv(filename):
    """Read a CSV from DATA_DIR and return a list of dict rows."""
    path = os.path.join(DATA_DIR, filename)
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
# Main analysis
# ===========================================================================
def main():
    print("Gulftrade Baseline Analysis")
    print(f"  As-of date        : {TODAY.date()}")
    print(f"  Lookback window   : {LOOKBACK_DAYS} days")
    print(f"  Forecast horizon  : {FORECAST_HORIZON_DAYS} days")
    print(f"  Inbound cutoff    : {INBOUND_CUTOFF.date()}")
    print(f"  Holding cost rate : {HOLDING_COST_RATE:.4%} of unit_cost / unit / day\n")

    # ------------------------------------------------------------------
    # 1. Load all CSVs
    # ------------------------------------------------------------------
    skus = load_csv("skus.csv")
    sales = load_csv("sales_history.csv")
    inventory = load_csv("inventory.csv")
    inbound = load_csv("inbound_orders.csv")

    # Build a lookup: sku_id → unit_cost
    unit_cost = {row["sku_id"]: float(row["unit_cost"]) for row in skus}

    # Build a lookup: (sku_id, location) → inventory row
    inv_lookup = {}
    for row in inventory:
        inv_lookup[(row["sku_id"], row["location"])] = row

    # ------------------------------------------------------------------
    # 2. Aggregate daily sales into a per (sku, location) dict
    #    keyed by date-string for easy look-up.
    # ------------------------------------------------------------------
    # sales_by_key[key][date_str] = units_sold
    sales_by_key = defaultdict(dict)
    for row in sales:
        key = (row["sku_id"], row["location"])
        sales_by_key[key][row["date"]] = int(row["units_sold"])

    # ------------------------------------------------------------------
    # 3. Determine the lookback window (last LOOKBACK_DAYS calendar days
    #    ending TODAY, inclusive).
    # ------------------------------------------------------------------
    lookback_dates = []
    for i in range(LOOKBACK_DAYS):
        d = TODAY - timedelta(days=LOOKBACK_DAYS - 1 - i)
        lookback_dates.append(d.strftime("%Y-%m-%d"))

    # ------------------------------------------------------------------
    # 4. Aggregate inbound quantities arriving within 30 days
    #    per (sku_id, destination).
    # ------------------------------------------------------------------
    # inbound_qty[key] = total units arriving on or before INBOUND_CUTOFF
    inbound_qty = defaultdict(int)
    for row in inbound:
        arrival = parse_date(row["expected_arrival"])
        if arrival <= INBOUND_CUTOFF:
            key = (row["sku_id"], row["destination"])
            inbound_qty[key] += int(row["quantity"])

    # ------------------------------------------------------------------
    # 5. For every (sku, location) pair found in the inventory file,
    #    compute forecast, effective stock, shortage, excess, holding cost.
    # ------------------------------------------------------------------
    results = []

    # Totals for the summary
    total_shortage = 0
    total_excess = 0
    total_holding_cost = 0.0

    for (sku_id, location), inv_row in sorted(inv_lookup.items()):
        # -- Forecast: average daily sales over the lookback window × 30 --
        # Include zero-sale days.  If a date is missing entirely from the
        # sales data (a planted gap), treat it as 0 units sold — same as
        # a stockout day.
        daily_sales = []
        for d_str in lookback_dates:
            units = sales_by_key.get((sku_id, location), {}).get(d_str, 0)
            daily_sales.append(units)

        avg_daily = sum(daily_sales) / len(daily_sales) if daily_sales else 0
        forecast_demand = avg_daily * FORECAST_HORIZON_DAYS

        # -- Effective stock --
        # available minus reserved gives net usable on-hand stock.
        available = int(inv_row["available"])
        reserved = int(inv_row["reserved"])
        on_hand = available - reserved

        # Add inbound arriving within the forecast horizon.
        inbound_units = inbound_qty.get((sku_id, location), 0)
        effective_stock = on_hand + inbound_units

        # -- Shortage / Excess --
        # Shortage: demand exceeds stock.  Excess: stock exceeds demand.
        # Both are floored at 0 — you can't be short and overstocked at once.
        shortage = max(0, forecast_demand - effective_stock)
        excess = max(0, effective_stock - forecast_demand)

        # -- Holding cost --
        # Applied to excess units for HOLDING_COST_PERIOD_DAYS days.
        # cost = excess_units × unit_cost × rate × period
        cost = unit_cost.get(sku_id, 0)
        holding_cost = excess * cost * HOLDING_COST_RATE * HOLDING_COST_PERIOD_DAYS

        results.append({
            "sku_id": sku_id,
            "location": location,
            "unit_cost": f"{cost:.2f}",
            "avg_daily_sales_last_14d": f"{avg_daily:.2f}",
            "forecast_demand_30d": f"{forecast_demand:.1f}",
            "available_stock": available,
            "reserved_stock": reserved,
            "inbound_within_30d": inbound_units,
            "effective_stock": effective_stock,
            "expected_shortage_units": int(shortage),
            "expected_excess_units": int(excess),
            "holding_cost_30d": f"{holding_cost:.2f}",
        })

        total_shortage += int(shortage)
        total_excess += int(excess)
        total_holding_cost += holding_cost

    # ------------------------------------------------------------------
    # 6. Save results to /output/baseline_results.csv
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
    fieldnames = [
        "sku_id",
        "location",
        "unit_cost",
        "avg_daily_sales_last_14d",
        "forecast_demand_30d",
        "available_stock",
        "reserved_stock",
        "inbound_within_30d",
        "effective_stock",
        "expected_shortage_units",
        "expected_excess_units",
        "holding_cost_30d",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print(f"Results written → {out_path}  ({len(results)} rows)\n")
    print("=" * 50)
    print(" BASELINE SUMMARY (no transfers)")
    print("=" * 50)
    print(f"  Total shortage units : {total_shortage:>8,}")
    print(f"  Total excess units   : {total_excess:>8,}")
    print(f"  Total holding cost   : ${total_holding_cost:>10,.2f}")
    print("=" * 50)

    # Highlight the hero case for quick reference
    print("\nHero case (SKU-001):")
    for r in results:
        if r["sku_id"] == "SKU-001":
            print(f"  {r['location']:<12}  "
                  f"forecast={r['forecast_demand_30d']:>7}  "
                  f"stock={r['effective_stock']:>5}  "
                  f"shortage={r['expected_shortage_units']:>5}  "
                  f"excess={r['expected_excess_units']:>5}")


if __name__ == "__main__":
    main()
