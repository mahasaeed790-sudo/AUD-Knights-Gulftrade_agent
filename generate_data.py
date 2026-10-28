#!/usr/bin/env python3
"""
Gulftrade Agent — Hackathon Data Generator
==========================================

Generates a synthetic supply-chain dataset as six CSV files inside ./data/.
All data is fictional. A fixed random seed makes output reproducible.

Usage:
    python3 generate_data.py
"""

import csv
import os
import random
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

LOCATIONS = ["Dubai", "Abu Dhabi", "Sharjah"]

# 90-day window ending "today" (fixed so output is deterministic)
END_DATE = datetime(2026, 10, 27)
START_DATE = END_DATE - timedelta(days=89)  # inclusive → 90 days
N_DAYS = (END_DATE - START_DATE).days + 1

# ---------------------------------------------------------------------------
# 1. SKUs  — 20 fictional products across 5 types
# ---------------------------------------------------------------------------
PRODUCT_TYPES = {
    "USB-C Charger": {
        "names": ["GulfCharge 20W", "GulfCharge 30W", "GulfCharge 65W", "GulfCharge 100W"],
        "cost_range": (6.0, 12.0),
    },
    "Cable": {
        "names": ["FlexLink USB-C 1m", "FlexLink USB-C 2m", "FlexLink Lightning 1m", "FlexLink HDMI 2m"],
        "cost_range": (1.5, 4.0),
    },
    "Power Bank": {
        "names": ["PowerCore 5000", "PowerCore 10000", "PowerCore 20000", "PowerCore 30000"],
        "cost_range": (8.0, 22.0),
    },
    "Keyboard": {
        "names": ["TypeFlow Mini", "TypeFlow Pro", "TypeFlow Mechanical", "TypeFlow Ergo"],
        "cost_range": (12.0, 45.0),
    },
    "Mouse": {
        "names": ["ClickWave Lite", "ClickWave Pro", "ClickWave Wireless", "ClickWave Ergo"],
        "cost_range": (5.0, 18.0),
    },
}


def build_skus():
    """Return list of dicts with sku_id, name, type, unit_cost."""
    skus = []
    idx = 1
    for ptype, info in PRODUCT_TYPES.items():
        for name in info["names"]:
            lo, hi = info["cost_range"]
            cost = round(random.uniform(lo, hi), 2)
            skus.append({
                "sku_id": f"SKU-{idx:03d}",
                "name": name,
                "type": ptype,
                "unit_cost": cost,
            })
            idx += 1
    return skus


# ---------------------------------------------------------------------------
# 2. Sales history  — 90 days per SKU per location
# ---------------------------------------------------------------------------
# Baseline daily demand per location (units). These are *reasonable* baselines;
# the hero case overrides specific SKU/location combos below.
BASELINE_DAILY_DEMAND = {
    "Dubai": 12,
    "Abu Dhabi": 8,
    "Sharjah": 6,
}

# Hero case: USB-C chargers SKU-001 .. SKU-004 with imbalanced stock vs demand.
# We give SKU-001 a strong mismatch:
#   stock  Dubai 500, Abu Dhabi 100, Sharjah 300
#   ~30-day demand  Dubai 200, Abu Dhabi 350, Sharjah 150
# Daily demand ≈ demand_30 / 30:
HERO_DAILY_DEMAND = {
    "SKU-001": {"Dubai": 7, "Abu Dhabi": 12, "Sharjah": 5},   # 210/360/150 over 30d
}
HERO_STOCK = {
    "SKU-001": {"Dubai": 500, "Abu Dhabi": 100, "Sharjah": 300},
}

# Stockout 1: SKU-007 (a cable) in Abu Dhabi for 10 days → sales drop to zero.
# day 77 = 2026-10-15; runs Oct 15–24 (10 days, ends 3 days before as-of).
# This stockout falls in the TEST / holdout period of the backtest.
STOCKOUT_SKU = "SKU-007"
STOCKOUT_LOCATION = "Abu Dhabi"
STOCKOUT_START_OFFSET = 77   # day index within the 90-day window (0-based)
STOCKOUT_DAYS = 10

# Stockout 2: SKU-013 (a keyboard) in Dubai for 8 days → sales drop to zero.
# day 42 = 2026-09-10; runs Sep 10–17 (8 days).
# This stockout falls in the TRAINING period of the backtest.
STOCKOUT2_SKU = "SKU-013"
STOCKOUT2_LOCATION = "Dubai"
STOCKOUT2_START_OFFSET = 42   # day index within the 90-day window (0-based)
STOCKOUT2_DAYS = 8

# All stockout definitions in one list for easy iteration.
STOCKOUTS = [
    (STOCKOUT_SKU, STOCKOUT_LOCATION, STOCKOUT_START_OFFSET, STOCKOUT_DAYS),
    (STOCKOUT2_SKU, STOCKOUT2_LOCATION, STOCKOUT2_START_OFFSET, STOCKOUT2_DAYS),
]

# Missing sales records: a few (sku, location, date) rows simply omitted.
# Deliberately sparse and scattered.
MISSING_RECORDS = [
    ("SKU-003", "Sharjah", 10),
    ("SKU-012", "Dubai", 33),
    ("SKU-015", "Abu Dhabi", 61),
    ("SKU-009", "Dubai", 77),
]


def build_sales_history(skus):
    """Return list of sales dicts and a set of omitted (sku,loc,date) keys."""
    rows = []
    omitted = set()
    for sku in skus:
        sid = sku["sku_id"]
        for loc in LOCATIONS:
            base = HERO_DAILY_DEMAND.get(sid, {}).get(loc, BASELINE_DAILY_DEMAND[loc])
            for d in range(N_DAYS):
                date_obj = START_DATE + timedelta(days=d)
                date_str = date_obj.strftime("%Y-%m-%d")

                # Missing record → skip entirely (row omitted from output)
                skip = False
                for msid, mloc, mday in MISSING_RECORDS:
                    if sid == msid and loc == mloc and d == mday:
                        omitted.add((sid, loc, date_str))
                        skip = True
                        break
                if skip:
                    continue

                # Stockout windows — check all planted stockouts
                in_stockout = False
                for s_sku, s_loc, s_start, s_days in STOCKOUTS:
                    if sid == s_sku and loc == s_loc:
                        if s_start <= d < s_start + s_days:
                            in_stockout = True
                            break
                if in_stockout:
                    # record the stockout as zero sales (row present, units 0)
                    rows.append({
                        "date": date_str,
                        "sku_id": sid,
                        "location": loc,
                        "units_sold": 0,
                    })
                    continue

                # normal noisy demand
                noise = random.gauss(0, max(1.0, base * 0.25))
                units = max(0, int(round(base + noise)))
                rows.append({
                    "date": date_str,
                    "sku_id": sid,
                    "location": loc,
                    "units_sold": units,
                })
    # sort for readability
    rows.sort(key=lambda r: (r["date"], r["sku_id"], r["location"]))
    return rows, omitted


# ---------------------------------------------------------------------------
# 3. Inventory  — current stock per SKU per location
# ---------------------------------------------------------------------------
SAFETY_STOCK_DEFAULT = 40


def build_inventory(skus):
    rows = []
    now = END_DATE
    for sku in skus:
        sid = sku["sku_id"]
        for loc in LOCATIONS:
            available = HERO_STOCK.get(sid, {}).get(loc, None)
            if available is None:
                available = random.randint(20, 200)

            # reserved stock (small fraction)
            reserved = random.randint(0, max(1, available // 5))

            # stock age: most under 120, a few deliberately over
            if sid in ("SKU-010", "SKU-014") and loc == "Sharjah":
                stock_age_days = random.randint(130, 210)   # aging stock
            elif sid == "SKU-018" and loc == "Dubai":
                stock_age_days = random.randint(130, 190)   # aging stock
            else:
                stock_age_days = random.randint(5, 110)

            safety = SAFETY_STOCK_DEFAULT
            if sid in HERO_DAILY_DEMAND:
                # hero SKU gets higher safety stock reference
                safety = 60

            rows.append({
                "sku_id": sid,
                "location": loc,
                "available": available,
                "reserved": reserved,
                "stock_age_days": stock_age_days,
                "safety_stock": safety,
                "last_updated": (now - timedelta(days=random.randint(0, 3))).strftime("%Y-%m-%d %H:%M:%S"),
            })
    rows.sort(key=lambda r: (r["sku_id"], r["location"]))
    return rows


# ---------------------------------------------------------------------------
# 4. Inbound orders  — expected arrivals + supplier lead time
# ---------------------------------------------------------------------------
def build_inbound_orders(skus):
    rows = []
    # one inbound order per SKU (some from Dubai port, some direct)
    for sku in skus:
        sid = sku["sku_id"]
        qty = random.randint(50, 400)
        lead_time = random.randint(7, 30)
        order_date = END_DATE - timedelta(days=random.randint(5, 20))
        expected_arrival = order_date + timedelta(days=lead_time)
        rows.append({
            "order_id": f"PO-{sid.split('-')[1]}",
            "sku_id": sid,
            "supplier": random.choice(["Delta Components", "MegaSource HK", "GulfParts FZE"]),
            "quantity": qty,
            "destination": random.choice(LOCATIONS),
            "order_date": order_date.strftime("%Y-%m-%d"),
            "expected_arrival": expected_arrival.strftime("%Y-%m-%d"),
            "supplier_lead_time_days": lead_time,
            "status": random.choice(["confirmed", "in_transit", "pending"]),
        })

    # Disruption-linked delay: SKU-002, SKU-003, SKU-004 (component shortage)
    for sid in ("SKU-002", "SKU-003", "SKU-004"):
        for row in rows:
            if row["sku_id"] == sid:
                row["status"] = "delayed"
                row["expected_arrival"] = (datetime.strptime(row["expected_arrival"], "%Y-%m-%d")
                                           + timedelta(days=21)).strftime("%Y-%m-%d")
                break

    rows.sort(key=lambda r: r["expected_arrival"])
    return rows


# ---------------------------------------------------------------------------
# 5. Transfer costs  — between each pair of locations
# ---------------------------------------------------------------------------
LOCATION_PAIRS = [
    ("Dubai", "Abu Dhabi"),
    ("Dubai", "Sharjah"),
    ("Abu Dhabi", "Sharjah"),
]


def build_transfer_costs():
    rows = []
    for src, dst in LOCATION_PAIRS:
        # symmetric-ish but we store both directions
        cost_per_unit = round(random.uniform(0.5, 3.0), 2)
        lead_time = random.randint(1, 3)
        rows.append({
            "from_location": src,
            "to_location": dst,
            "cost_per_unit": cost_per_unit,
            "lead_time_days": lead_time,
        })
        rows.append({
            "from_location": dst,
            "to_location": src,
            "cost_per_unit": cost_per_unit,
            "lead_time_days": lead_time,
        })

    # PLANTED BAD TRANSFER: Abu Dhabi → Dubai for SKU-001 hero case.
    # Hero: Abu Dhabi stock 100 vs 30-day demand 360 (short ~260).
    #       Dubai stock 500 vs 30-day demand 210 (surplus ~290).
    # So the *sensible* transfer is Dubai→Abu Dhabi, NOT Abu Dhabi→Dubai.
    # We inflate the Abu Dhabi→Dubai cost so that transferring from the
    # stockout-risk location to the surplus location costs more than any
    # benefit. Using unit_cost ~ $9 for SKU-001, transferring 100 units
    # the wrong way "costs" 100*9 = $900 of lost sales in Abu Dhabi plus
    # the transfer fee, vs negligible benefit in Dubai.
    for row in rows:
        if row["from_location"] == "Abu Dhabi" and row["to_location"] == "Dubai":
            row["cost_per_unit"] = 12.50  # deliberately high
            row["lead_time_days"] = 5     # deliberately slow
    return rows


# ---------------------------------------------------------------------------
# 6. Disruptions  — two dated events
# ---------------------------------------------------------------------------
def build_disruptions():
    shortage_date = END_DATE - timedelta(days=15)
    route_date = END_DATE - timedelta(days=8)
    rows = [
        {
            "disruption_id": "DIS-001",
            "date": shortage_date.strftime("%Y-%m-%d"),
            "type": "component_shortage",
            "description": "Shortage of USB-C controller ICs affecting charger production.",
            "affected_skus": "SKU-002;SKU-003;SKU-004",
            "affected_locations": "Dubai;Abu Dhabi;Sharjah",
            "severity": "high",
            "status": "ongoing",
        },
        {
            "disruption_id": "DIS-002",
            "date": route_date.strftime("%Y-%m-%d"),
            "type": "shipping_route_delay",
            "description": "Port congestion at Jebel Ali delaying inbound containers 5-10 days.",
            "affected_skus": "ALL",
            "affected_locations": "Dubai",
            "severity": "medium",
            "status": "ongoing",
        },
    ]
    return rows


# ---------------------------------------------------------------------------
# CSV writer helper
# ---------------------------------------------------------------------------
def write_csv(filename, fieldnames, rows):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):>5} rows → {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Gulftrade data generator  (seed={RANDOM_SEED})")
    print(f"Output directory: {DATA_DIR}")
    print(f"Date window: {START_DATE.date()} → {END_DATE.date()}  ({N_DAYS} days)\n")

    skus = build_skus()
    write_csv("skus.csv", ["sku_id", "name", "type", "unit_cost"], skus)

    sales, omitted = build_sales_history(skus)
    write_csv("sales_history.csv", ["date", "sku_id", "location", "units_sold"], sales)

    inv = build_inventory(skus)
    write_csv("inventory.csv",
              ["sku_id", "location", "available", "reserved", "stock_age_days", "safety_stock", "last_updated"],
              inv)

    inbound = build_inbound_orders(skus)
    write_csv("inbound_orders.csv",
              ["order_id", "sku_id", "supplier", "quantity", "destination",
               "order_date", "expected_arrival", "supplier_lead_time_days", "status"],
              inbound)

    transfers = build_transfer_costs()
    write_csv("transfer_costs.csv", ["from_location", "to_location", "cost_per_unit", "lead_time_days"], transfers)

    disruptions = build_disruptions()
    write_csv("disruptions.csv",
              ["disruption_id", "date", "type", "description", "affected_skus",
               "affected_locations", "severity", "status"],
              disruptions)

    print(f"\nDone. {len(omitted)} sales records intentionally omitted (missing data).")
    print("See answer_key.md for the full list of planted problems.")


if __name__ == "__main__":
    main()
