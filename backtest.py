#!/usr/bin/env python3
"""
Gulftrade Agent — Forecast Backtest
====================================

Step 3c of the hackathon pipeline.

This script splits the 90-day sales history into a 60-day training set and
a 30-day holdout (test) set, produces two forecasts on the training data
only, and measures accuracy against actual sales in the holdout period.

Two forecasts compared:
  a. **Baseline** — average daily sales over the last 14 training days × 30
     (includes zero-sales days — the naive method from baseline.py).
  b. **Smart** — the stockout-corrected EWMA method from forecast.py
     (detect stockouts, fill with average demand, apply exponential weighting).

Accuracy metrics:
  - **WAPE** (Weighted Absolute Percentage Error) = Σ|forecast − actual| / Σ actual
  - **Bias** = mean(forecast − actual)  (positive = over-forecast, negative = under)

Usage:
    python3 backtest.py
"""

import csv
import os
from collections import defaultdict
from datetime import datetime, timedelta

# Reuse the forecast engine from forecast.py — do not copy-paste.
from forecast import (
    detect_stockout_days,
    detect_missing_days,
    compute_ewma,
    EWMA_ALPHA,
    STOCKOUT_RUN_THRESHOLD,
)

# ===========================================================================
# ASSUMPTIONS — change these to tune the backtest
# ===========================================================================

# "As of" date — end of the 90-day data window.
TODAY = datetime(2026, 10, 27)

# Total history length and split point.
HISTORY_DAYS = 90
TRAINING_DAYS = 60     # first 60 days → training
TEST_DAYS = 30         # last 30 days → holdout / actuals

# Derived: the start of the full window.
HISTORY_START = TODAY - timedelta(days=HISTORY_DAYS - 1)

# Training window: HISTORY_START .. HISTORY_START + TRAINING_DAYS - 1
TRAIN_START = HISTORY_START
TRAIN_END = HISTORY_START + timedelta(days=TRAINING_DAYS - 1)

# Test window: TRAIN_END + 1 .. TODAY
TEST_START = TRAIN_END + timedelta(days=1)
TEST_END = TODAY

# Baseline lookback: last N days of the *training* period.
BASELINE_LOOKBACK_DAYS = 14

# Forecast horizon (must match the test window length for comparison).
FORECAST_HORIZON_DAYS = TEST_DAYS   # 30

# ===========================================================================
# Paths
# ===========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# ===========================================================================
# CSV loading helper
# ===========================================================================
def load_csv(filename):
    """Read a CSV from DATA_DIR and return a list of dict rows."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ===========================================================================
# Date helpers
# ===========================================================================
def date_range_strs(start, end):
    """Return a list of YYYY-MM-DD strings from start to end (inclusive)."""
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


# ===========================================================================
# Forecast methods (operate on the training window only)
# ===========================================================================
def forecast_baseline(train_series):
    """
    Naive baseline forecast.

    Takes the last 14 days of the training series (including any zero-sales
    days), computes the average daily sales, and multiplies by 30.

    train_series: list of (date_str, units_or_None) for the 60 training days.
    Returns: float (30-day demand forecast)
    """
    # Take the last BASELINE_LOOKBACK_DAYS entries from the training series.
    lookback = train_series[-BASELINE_LOOKBACK_DAYS:]

    # Sum daily sales, treating missing days (None) as 0.
    total = sum(u if u is not None else 0 for _, u in lookback)
    avg_daily = total / len(lookback)
    return avg_daily * FORECAST_HORIZON_DAYS


def forecast_smart(train_series):
    """
    Stockout-corrected EWMA forecast (same method as forecast.py).

    Steps:
      1. Detect stockout days (runs of 3+ consecutive zero/missing days).
      2. Detect missing days (not part of a stockout run).
      3. Fill excluded days with the average of remaining "normal" days.
      4. Apply EWMA over the cleaned 60-day series.
      5. Multiply the final EWMA value by 30 for the 30-day forecast.

    train_series: list of (date_str, units_or_None) for the 60 training days.
    Returns: (float forecast, set of stockout dates, set of missing dates)
    """
    # Step 1: detect stockout days in the training window.
    stockout_days = detect_stockout_days(train_series)

    # Step 2: detect missing days (not already flagged as stockout).
    missing_days = detect_missing_days(train_series)
    missing_days -= stockout_days

    # Step 3: compute the average of "normal" (non-excluded) days.
    normal_values = []
    for d_str, units in train_series:
        if d_str in stockout_days or d_str in missing_days:
            continue
        normal_values.append(units)
    fill_value = sum(normal_values) / len(normal_values) if normal_values else 0.0

    # Build the cleaned series: real values for normal days, fill for excluded.
    cleaned = []
    for d_str, units in train_series:
        if d_str in stockout_days or d_str in missing_days:
            cleaned.append(fill_value)
        else:
            cleaned.append(units)

    # Step 4: EWMA over the cleaned series.
    daily_ewma = compute_ewma(cleaned, EWMA_ALPHA)

    # Step 5: scale to 30 days.
    forecast = daily_ewma * FORECAST_HORIZON_DAYS
    return forecast, stockout_days, missing_days


# ===========================================================================
# Accuracy metrics
# ===========================================================================
def compute_wape(forecasts, actuals):
    """
    WAPE (Weighted Absolute Percentage Error).

    WAPE = Σ |forecast_i − actual_i|  /  Σ actual_i

    Unlike MAPE, WAPE doesn't blow up when individual actuals are zero.
    It weights each row's error by its share of total actual demand.
    """
    abs_errors = sum(abs(f - a) for f, a in zip(forecasts, actuals))
    total_actual = sum(actuals)
    if total_actual == 0:
        return float("inf")
    return abs_errors / total_actual


def compute_bias(forecasts, actuals):
    """
    Bias = mean(forecast − actual).

    Positive → forecasts tend to be too high (over-forecast).
    Negative → forecasts tend to be too low (under-forecast).
    """
    if not forecasts:
        return 0.0
    return sum(f - a for f, a in zip(forecasts, actuals)) / len(forecasts)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Gulftrade Forecast Backtest")
    print(f"  As-of date        : {TODAY.date()}")
    print(f"  Training window   : {TRAIN_START.date()} → {TRAIN_END.date()} ({TRAINING_DAYS} days)")
    print(f"  Test / holdout    : {TEST_START.date()} → {TEST_END.date()} ({TEST_DAYS} days)")
    print(f"  Baseline lookback : {BASELINE_LOOKBACK_DAYS} training days")
    print(f"  EWMA alpha        : {EWMA_ALPHA}")
    print(f"  Stockout threshold: {STOCKOUT_RUN_THRESHOLD}+ consecutive zeros\n")

    # ------------------------------------------------------------------
    # 1. Load sales history
    # ------------------------------------------------------------------
    sales = load_csv("sales_history.csv")

    # Build per (sku, location) → {date_str: units_sold}
    sales_map = defaultdict(dict)
    for row in sales:
        sales_map[(row["sku_id"], row["location"])][row["date"]] = int(row["units_sold"])

    # ------------------------------------------------------------------
    # 2. Build the date indices for training and test windows
    # ------------------------------------------------------------------
    train_dates = date_range_strs(TRAIN_START, TRAIN_END)
    test_dates = date_range_strs(TEST_START, TEST_END)

    # ------------------------------------------------------------------
    # 3. For each (sku, location), build forecasts and compare to actuals
    # ------------------------------------------------------------------
    results = []

    # Lists to accumulate forecasts and actuals for aggregate metrics.
    all_baseline_fc = []
    all_smart_fc = []
    all_actuals = []

    # Subset: rows where training contained a stockout (3+ zero run).
    stockout_train_baseline_fc = []
    stockout_train_smart_fc = []
    stockout_train_actuals = []

    # Subset: rows where training contained missing records.
    missing_train_baseline_fc = []
    missing_train_smart_fc = []
    missing_train_actuals = []

    # Subset: all rows EXCLUDING those whose test period contains suppressed
    # (stockout) days.  This gives a "clean" WAPE that isn't polluted by
    # artificially low actuals caused by stockouts in the holdout period.
    clean_baseline_fc = []
    clean_smart_fc = []
    clean_actuals = []

    for key in sorted(sales_map.keys()):
        sku_id, location = key
        date_units = sales_map[key]

        # -- Build training series: (date_str, units_or_None) for 60 days --
        train_series = []
        for d_str in train_dates:
            units = date_units.get(d_str)  # None if missing
            train_series.append((d_str, units))

        # -- Build test actuals: sum of units_sold in the 30 holdout days --
        # Missing test days are treated as 0 actual sales (data gap).
        test_units = []
        test_stockout_days = []  # dates in test where sales are suppressed
        for d_str in test_dates:
            units = date_units.get(d_str)
            if units is None:
                # Missing record in test — treat as 0 for actuals.
                test_units.append(0)
            else:
                test_units.append(units)
                # Track zero-sale days in test (potential stockout suppression).
                if units == 0:
                    test_stockout_days.append(d_str)

        actual_30d = sum(test_units)

        # -- Baseline forecast (naive, last 14 training days × 30) --
        baseline_fc = forecast_baseline(train_series)

        # -- Smart forecast (stockout-corrected EWMA) --
        smart_fc, train_stockout_days, train_missing_days = forecast_smart(train_series)

        # -- Per-row error metrics --
        baseline_abs_err = abs(baseline_fc - actual_30d)
        smart_abs_err = abs(smart_fc - actual_30d)

        # -- Flag: did training contain a stockout? --
        had_train_stockout = len(train_stockout_days) > 0
        had_train_missing = len(train_missing_days) > 0

        # -- Flag: does test contain suppressed-sales days? --
        # A run of 3+ zero-sale days in test means actuals are artificially low.
        test_series = [(d_str, date_units.get(d_str)) for d_str in test_dates]
        test_stockout_set = detect_stockout_days(test_series)
        has_test_stockout = len(test_stockout_set) > 0

        # -- Build a plain-English note --
        notes = []
        if had_train_stockout:
            notes.append(f"training stockout: {len(train_stockout_days)} days adjusted")
        if had_train_missing:
            notes.append(f"training missing: {len(train_missing_days)} days filled")
        if has_test_stockout:
            notes.append(f"test stockout: {len(test_stockout_set)} days suppressed actuals")
        note = "; ".join(notes) if notes else "no issues"

        results.append({
            "sku_id": sku_id,
            "location": location,
            "baseline_forecast": f"{baseline_fc:.1f}",
            "smart_forecast": f"{smart_fc:.1f}",
            "actual_30d": actual_30d,
            "baseline_abs_error": f"{baseline_abs_err:.1f}",
            "smart_abs_error": f"{smart_abs_err:.1f}",
            "train_stockout_days": len(train_stockout_days),
            "train_missing_days": len(train_missing_days),
            "test_stockout_days": len(test_stockout_set),
            "had_train_stockout": "yes" if had_train_stockout else "no",
            "had_train_missing": "yes" if had_train_missing else "no",
            "has_test_stockout": "yes" if has_test_stockout else "no",
            "note": note,
        })

        # Accumulate for overall metrics.
        all_baseline_fc.append(baseline_fc)
        all_smart_fc.append(smart_fc)
        all_actuals.append(actual_30d)

        # Subset: training had a stockout.
        if had_train_stockout:
            stockout_train_baseline_fc.append(baseline_fc)
            stockout_train_smart_fc.append(smart_fc)
            stockout_train_actuals.append(actual_30d)

        # Subset: training had missing records.
        if had_train_missing:
            missing_train_baseline_fc.append(baseline_fc)
            missing_train_smart_fc.append(smart_fc)
            missing_train_actuals.append(actual_30d)

        # Subset: clean rows — exclude rows whose test period has suppressed
        # (stockout) days so WAPE is not polluted by artificially low actuals.
        if not has_test_stockout:
            clean_baseline_fc.append(baseline_fc)
            clean_smart_fc.append(smart_fc)
            clean_actuals.append(actual_30d)

    # ------------------------------------------------------------------
    # 4. Compute aggregate metrics
    # ------------------------------------------------------------------
    overall_wape_baseline = compute_wape(all_baseline_fc, all_actuals)
    overall_wape_smart = compute_wape(all_smart_fc, all_actuals)
    overall_bias_baseline = compute_bias(all_baseline_fc, all_actuals)
    overall_bias_smart = compute_bias(all_smart_fc, all_actuals)

    stockout_wape_baseline = compute_wape(stockout_train_baseline_fc, stockout_train_actuals) if stockout_train_actuals else None
    stockout_wape_smart = compute_wape(stockout_train_smart_fc, stockout_train_actuals) if stockout_train_actuals else None
    stockout_bias_baseline = compute_bias(stockout_train_baseline_fc, stockout_train_actuals) if stockout_train_actuals else None
    stockout_bias_smart = compute_bias(stockout_train_smart_fc, stockout_train_actuals) if stockout_train_actuals else None

    missing_wape_baseline = compute_wape(missing_train_baseline_fc, missing_train_actuals) if missing_train_actuals else None
    missing_wape_smart = compute_wape(missing_train_smart_fc, missing_train_actuals) if missing_train_actuals else None
    missing_bias_baseline = compute_bias(missing_train_baseline_fc, missing_train_actuals) if missing_train_actuals else None
    missing_bias_smart = compute_bias(missing_train_smart_fc, missing_train_actuals) if missing_train_actuals else None

    # Clean subset: excludes rows with stockout-suppressed actuals in the test period.
    clean_wape_baseline = compute_wape(clean_baseline_fc, clean_actuals) if clean_actuals else None
    clean_wape_smart = compute_wape(clean_smart_fc, clean_actuals) if clean_actuals else None
    clean_bias_baseline = compute_bias(clean_baseline_fc, clean_actuals) if clean_actuals else None
    clean_bias_smart = compute_bias(clean_smart_fc, clean_actuals) if clean_actuals else None

    # ------------------------------------------------------------------
    # 5. Save per-row results
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "backtest_results.csv")
    fieldnames = [
        "sku_id",
        "location",
        "baseline_forecast",
        "smart_forecast",
        "actual_30d",
        "baseline_abs_error",
        "smart_abs_error",
        "train_stockout_days",
        "train_missing_days",
        "test_stockout_days",
        "had_train_stockout",
        "had_train_missing",
        "has_test_stockout",
        "note",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    print(f"Results written → {out_path}  ({len(results)} rows)\n")

    def fmt_wape(v):
        return f"{v*100:.1f}%" if v is not None and v != float("inf") else "N/A"

    def fmt_bias(v):
        return f"{v:+.1f}" if v is not None else "N/A"

    print("=" * 60)
    print(" BACKTEST SUMMARY — WAPE & Bias")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>12} {'Smart':>12}")
    print("-" * 60)
    print(f"{'Overall WAPE':<30} {fmt_wape(overall_wape_baseline):>12} {fmt_wape(overall_wape_smart):>12}")
    print(f"{'Overall bias (units)':<30} {fmt_bias(overall_bias_baseline):>12} {fmt_bias(overall_bias_smart):>12}")
    print("-" * 60)
    print(f"{'Stockout-in-training WAPE':<30} {fmt_wape(stockout_wape_baseline):>12} {fmt_wape(stockout_wape_smart):>12}")
    print(f"{'Stockout-in-training bias':<30} {fmt_bias(stockout_bias_baseline):>12} {fmt_bias(stockout_bias_smart):>12}")
    print(f"{'  (rows in this subset)':<30} {len(stockout_train_actuals):>12} {len(stockout_train_actuals):>12}")
    print("-" * 60)
    print(f"{'Missing-records-in-training WAPE':<30} {fmt_wape(missing_wape_baseline):>12} {fmt_wape(missing_wape_smart):>12}")
    print(f"{'Missing-records-in-training bias':<30} {fmt_bias(missing_bias_baseline):>12} {fmt_bias(missing_bias_smart):>12}")
    print(f"{'  (rows in this subset)':<30} {len(missing_train_actuals):>12} {len(missing_train_actuals):>12}")
    print("-" * 60)
    print(f"{'Clean WAPE (excl. test stockouts)':<30} {fmt_wape(clean_wape_baseline):>12} {fmt_wape(clean_wape_smart):>12}")
    print(f"{'Clean bias (excl. test stockouts)':<30} {fmt_bias(clean_bias_baseline):>12} {fmt_bias(clean_bias_smart):>12}")
    print(f"{'  (rows in this subset)':<30} {len(clean_actuals):>12} {len(clean_actuals):>12}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 7. Highlight SKU-007 Abu Dhabi (stockout is in the test period)
    # ------------------------------------------------------------------
    print("\nSKU-007 Abu Dhabi (stockout Oct 15–24 falls in TEST period):")
    print("-" * 60)
    for r in results:
        if r["sku_id"] == "SKU-007" and r["location"] == "Abu Dhabi":
            print(f"  Baseline forecast : {r['baseline_forecast']:>8}")
            print(f"  Smart forecast    : {r['smart_forecast']:>8}")
            print(f"  Actual (30d)      : {r['actual_30d']:>8}")
            print(f"  Baseline abs error: {r['baseline_abs_error']:>8}")
            print(f"  Smart abs error   : {r['smart_abs_error']:>8}")
            print(f"  Train stockout days : {r['train_stockout_days']}")
            print(f"  Train missing days  : {r['train_missing_days']}")
            print(f"  Test stockout days  : {r['test_stockout_days']}")
            print(f"  Note: {r['note']}")
            print()
            print("  Note: The stockout is in the TEST period, so actual sales")
            print("  are suppressed (10 zero-days). Both forecasts were built on")
            print("  clean training data, so they are similar — the large error")
            print("  vs. actuals is because actuals are artificially low, not")
            print("  because the forecast is bad.")
            break
    print("-" * 60)

    # ------------------------------------------------------------------
    # 8. Plain-English verdict
    # ------------------------------------------------------------------
    # Use the CLEAN WAPE (excluding test-period stockouts) as the primary
    # comparison, since it's not polluted by artificially low actuals.
    # Also mention the overall WAPE for completeness.
    print()
    clean_diff = clean_wape_baseline - clean_wape_smart  # positive = smart is better
    if clean_wape_smart < clean_wape_baseline:
        pct_rel = (clean_diff / clean_wape_baseline) * 100
        print(f"VERDICT: The smart forecast beat the baseline on the clean subset "
              f"(excluding test-period stockouts), reducing WAPE from "
              f"{clean_wape_baseline*100:.1f}% to {clean_wape_smart*100:.1f}% "
              f"(a {pct_rel:.1f}% relative improvement). Overall WAPE including "
              f"suppressed rows was {overall_wape_baseline*100:.1f}% (baseline) vs "
              f"{overall_wape_smart*100:.1f}% (smart).")
    elif clean_wape_smart > clean_wape_baseline:
        pct_rel = ((clean_wape_smart - clean_wape_baseline) / clean_wape_baseline) * 100
        print(f"VERDICT: The smart forecast did NOT beat the baseline on the clean "
              f"subset — WAPE went from {clean_wape_baseline*100:.1f}% (baseline) to "
              f"{clean_wape_smart*100:.1f}% (smart), a {pct_rel:.1f}% relative "
              f"worsening. Overall WAPE including suppressed rows was "
              f"{overall_wape_baseline*100:.1f}% (baseline) vs "
              f"{overall_wape_smart*100:.1f}% (smart).")
    else:
        print(f"VERDICT: The smart forecast tied the baseline on the clean subset "
              f"(WAPE = {clean_wape_baseline*100:.1f}% for both). Overall WAPE "
              f"including suppressed rows was {overall_wape_baseline*100:.1f}% "
              f"vs {overall_wape_smart*100:.1f}%.")
    print()


if __name__ == "__main__":
    main()
