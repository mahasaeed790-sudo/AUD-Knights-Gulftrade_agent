#!/usr/bin/env python3
"""
Gulftrade Agent — Streamlit Decision-Support App
=================================================

Wraps engine.py into an interactive planner UI.  All computation is done
by engine.py's real functions and State object — this file only handles
presentation, session state, and user interaction.

Run:
    streamlit run app.py
"""

import csv
import os
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Import everything we need from engine.py — no duplicated logic.
import engine
from engine import (
    State,
    run_engine,
    AS_OF_DATE,
    FORECAST_HORIZON_DAYS,
    UNIT_MARGIN_RATE,
    AGEING_THRESHOLD_DAYS,
    COMPONENT_SHORTAGE_DELAY_DAYS,
    SHIPPING_ROUTE_DELAY_DAYS,
    OUTPUT_DIR,
)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def init_session():
    """Initialise all session-state keys on first load or reset."""
    st.session_state["disruption_toggles"] = {}
    # Default: all disruptions ON (matches engine.py default).
    for d in State().disruptions:
        st.session_state["disruption_toggles"][d["disruption_id"]] = True
    # Inventory edits: {(sku, loc): {field: new_value, ...}}
    st.session_state["inventory_edits"] = {}
    # Approval statuses keyed by rec_key = (sku, from, to, qty)
    st.session_state["approval_status"] = {}
    # Audit log entries: list of dicts {timestamp, action, detail}
    st.session_state["audit_log"] = []


def ensure_session():
    """Make sure session state is initialised."""
    if "disruption_toggles" not in st.session_state:
        init_session()


def audit(action, detail):
    """Append an entry to the in-memory audit log."""
    st.session_state["audit_log"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "detail": detail,
    })


def save_audit_log():
    """Write the audit log to output/audit_log.csv."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "audit_log.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "action", "detail"])
        writer.writeheader()
        writer.writerows(st.session_state["audit_log"])
    return path


def build_state():
    """
    Build a fresh engine.State, apply the current sidebar toggles, then
    apply any planner inventory edits.  Returns the ready-to-run state.
    """
    state = State()
    for did, active in st.session_state["disruption_toggles"].items():
        state.toggle_disruption(did, active)

    # Apply inventory edits from the planner.
    for (sku, loc), fields in st.session_state["inventory_edits"].items():
        for field, value in fields.items():
            state.edit_inventory(sku, loc, field, value)

    return state


def rec_key(rec):
    """Stable key for a recommendation row."""
    return (rec["sku_id"], rec["from_location"], rec["to_location"], int(rec["quantity"]))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Gulftrade Transfer Engine",
        page_icon="📦",
        layout="wide",
    )

    ensure_session()

    # --- Top banner ---
    st.markdown(
        '<div style="background:#fff3cd;color:#856404;padding:10px 18px;'
        'border-radius:6px;font-weight:600;text-align:center;margin-bottom:8px;">'
        'Synthetic data. Decision support only: no transfers are executed.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.title("Gulftrade Transfer Engine")
    st.caption(
        f"As-of date: {AS_OF_DATE.date()}  |  Forecast horizon: "
        f"{FORECAST_HORIZON_DAYS} days  |  Unit margin: "
        f"{UNIT_MARGIN_RATE:.0%} of unit_cost (ASSUMPTION)  |  "
        f"Ageing threshold: {AGEING_THRESHOLD_DAYS} days"
    )

    # --- Sidebar ---
    render_sidebar()

    # --- Build state + run engine ---
    state = build_state()
    results = run_engine(state)

    # --- Tabs ---
    tab_inv, tab_fc, tab_imb, tab_rec, tab_imp = st.tabs(
        ["Inventory", "Forecast", "Imbalances", "Recommendations", "Impact"]
    )

    with tab_inv:
        render_inventory_tab(state, results)

    with tab_fc:
        render_forecast_tab(state, results)

    with tab_imb:
        render_imbalances_tab(results)

    with tab_rec:
        render_recommendations_tab(state, results)

    with tab_imp:
        render_impact_tab(state, results)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.header("Disruptions")

    # Load disruptions for metadata display.
    disruptions = State().disruptions

    for d in disruptions:
        did = d["disruption_id"]
        dtype = d["type"]
        is_on = st.session_state["disruption_toggles"].get(did, True)

        with st.sidebar.expander(f"{did} — {dtype}", expanded=True):
            st.write(f"**Date:** {d['date']}")
            st.write(f"**Source:** {d['description']}")

            if dtype == "component_shortage":
                skus = d["affected_skus"]
                st.write(f"**Affected SKUs:** {skus}")
                st.write(f"**Affected locations:** {d['affected_locations']}")
                st.write(f"**Delay applied:** +{COMPONENT_SHORTAGE_DELAY_DAYS} days to inbound POs")
            elif dtype == "shipping_route_delay":
                st.write(f"**Affected SKUs:** {d['affected_skus']}")
                st.write(f"**Affected locations:** {d['affected_locations']}")
                st.write(f"**Delay applied:** +{SHIPPING_ROUTE_DELAY_DAYS} days to transfers & inbound")

            new_val = st.checkbox(
                "Active",
                value=is_on,
                key=f"toggle_{did}",
                help="Toggle this disruption on or off.",
            )
            if new_val != st.session_state["disruption_toggles"].get(did, True):
                st.session_state["disruption_toggles"][did] = new_val
                audit(
                    "Disruption toggle",
                    f"{did} ({dtype}) → {'ON' if new_val else 'OFF'}",
                )
                st.rerun()

    st.sidebar.divider()

    # Reset button.
    if st.sidebar.button("Reset data", type="primary"):
        init_session()
        audit("Reset", "All toggles, edits, and approvals reset to defaults.")
        st.rerun()

    st.sidebar.divider()

    # Show edit / approval counts.
    n_edits = len(st.session_state["inventory_edits"])
    n_approved = sum(1 for v in st.session_state["approval_status"].values() if v == "approved")
    n_rejected = sum(1 for v in st.session_state["approval_status"].values() if v == "rejected")
    st.sidebar.caption(
        f"Planner edits: {n_edits}  |  Approved: {n_approved}  |  Rejected: {n_rejected}"
    )


# ---------------------------------------------------------------------------
# Tab 1: Inventory
# ---------------------------------------------------------------------------
def render_inventory_tab(state, results):
    st.header("Inventory")

    # SKU filter.
    all_skus = sorted(set(r["sku_id"] for r in state.inventory))
    sku_filter = st.selectbox("Filter by SKU", ["All"] + all_skus, key="inv_sku_filter")

    # Build a display DataFrame from the state's inventory rows.
    rows = []
    for key in state.all_keys:
        sku_id, location = key
        inv = state.inv_lookup[key]

        # Check for planner edits.
        edit = st.session_state["inventory_edits"].get(key, {})
        edited = "edited by planner" if edit else ""

        available = int(inv["available"])
        reserved = int(inv["reserved"])
        stock_age = int(inv["stock_age_days"])

        # Inbound within 30d from engine.
        inbound = engine.get_inbound_qty(state, sku_id, location)

        rows.append({
            "SKU": sku_id,
            "Location": location,
            "Available": available,
            "Reserved": reserved,
            "Incoming (30d)": inbound,
            "Effective": available - reserved + inbound,
            "Safety stock": int(inv["safety_stock"]),
            "Stock age (days)": stock_age,
            "Last updated": inv.get("last_updated", ""),
            "Source file": "data/inventory.csv",
            "Status": edited,
        })

    df = pd.DataFrame(rows)
    if sku_filter != "All":
        df = df[df["SKU"] == sku_filter].reset_index(drop=True)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Inline editing ---
    st.subheader("Edit a record")
    st.caption("Changes are stored in session state and marked 'edited by planner'. "
               "All downstream tabs recompute automatically.")

    edit_skus = sorted(df["SKU"].unique()) if len(df) > 0 else all_skus
    col1, col2 = st.columns(2)
    with col1:
        edit_sku = st.selectbox("SKU", edit_skus, key="edit_sku")
    with col2:
        # Locations for the selected SKU.
        sku_locs = sorted(df[df["SKU"] == edit_sku]["Location"].unique()) if len(df) > 0 else []
        edit_loc = st.selectbox("Location", sku_locs, key="edit_loc")

    if edit_sku and edit_loc:
        key = (edit_sku, edit_loc)
        inv = state.inv_lookup.get(key, {})
        current_avail = int(inv.get("available", 0))
        current_res = int(inv.get("reserved", 0))
        current_age = int(inv.get("stock_age_days", 0))

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            new_avail = st.number_input("Available", value=current_avail, min_value=0, key="edit_avail")
        with col_b:
            new_res = st.number_input("Reserved", value=current_res, min_value=0, key="edit_res")
        with col_c:
            new_age = st.number_input("Stock age (days)", value=current_age, min_value=0, key="edit_age")

        if st.button("Apply edit", key="apply_edit"):
            st.session_state["inventory_edits"][key] = {
                "available": str(new_avail),
                "reserved": str(new_res),
                "stock_age_days": str(new_age),
            }
            audit(
                "Inventory edit",
                f"{edit_sku} @ {edit_loc}: available {current_avail}→{new_avail}, "
                f"reserved {current_res}→{new_res}, age {current_age}→{new_age}",
            )
            st.success(f"Updated {edit_sku} @ {edit_loc}. All tabs will recompute.")
            st.rerun()

    # Show current edits.
    if st.session_state["inventory_edits"]:
        st.subheader("Applied edits")
        edit_rows = []
        for (sku, loc), fields in st.session_state["inventory_edits"].items():
            edit_rows.append({
                "SKU": sku,
                "Location": loc,
                "Available": fields.get("available", ""),
                "Reserved": fields.get("reserved", ""),
                "Stock age": fields.get("stock_age_days", ""),
            })
        st.dataframe(pd.DataFrame(edit_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: Forecast
# ---------------------------------------------------------------------------
def render_forecast_tab(state, results):
    st.header("Forecast (stockout-corrected EWMA)")

    all_skus = sorted(set(k[0] for k in state.all_keys))
    selected_sku = st.selectbox("Choose SKU", all_skus, key="fc_sku")

    # Gather forecast + stock data for the selected SKU across locations.
    chart_data = []
    table_rows = []
    for key in state.all_keys:
        sku_id, location = key
        if sku_id != selected_sku:
            continue
        fc = results["forecasts"].get(key, {})
        inv = state.inv_lookup[key]
        available = int(inv["available"])
        reserved = int(inv["reserved"])
        inbound = engine.get_inbound_qty(state, sku_id, location)
        eff_stock = available - reserved + inbound

        forecast_30d = fc.get("forecast_30d", 0)
        forecast_low = fc.get("forecast_low", 0)
        forecast_high = fc.get("forecast_high", 0)
        dq_flag = fc.get("data_quality_flag", "ok")
        note = fc.get("note", "")

        table_rows.append({
            "Location": location,
            "Forecast 30d": f"{forecast_30d:.1f}",
            "P10 (low)": f"{forecast_low:.1f}",
            "P90 (high)": f"{forecast_high:.1f}",
            "Effective stock": eff_stock,
            "Safety stock": int(inv["safety_stock"]),
            "Data quality": dq_flag,
            "Note": note,
        })

        chart_data.append({
            "location": location,
            "forecast_30d": forecast_30d,
            "forecast_low": forecast_low,
            "forecast_high": forecast_high,
            "effective_stock": eff_stock,
        })

    if not table_rows:
        st.write("No data for this SKU.")
        return

    # Table.
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Chart: forecast range vs stock, grouped by location.
    fig = go.Figure()
    locations = [d["location"] for d in chart_data]

    # P10–P90 range as a filled area.
    fig.add_trace(go.Scatter(
        x=locations,
        y=[d["forecast_low"] for d in chart_data],
        name="P10 (low)",
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=locations,
        y=[d["forecast_high"] for d in chart_data],
        name="P10–P90 range",
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(100,149,237,0.25)",
    ))

    # Forecast 30d line.
    fig.add_trace(go.Scatter(
        x=locations,
        y=[d["forecast_30d"] for d in chart_data],
        name="Forecast 30d",
        mode="lines+markers",
        line=dict(color="royalblue", width=2),
    ))

    # Effective stock bars.
    fig.add_trace(go.Bar(
        x=locations,
        y=[d["effective_stock"] for d in chart_data],
        name="Effective stock",
        marker_color="rgba(60,179,113,0.6)",
    ))

    fig.update_layout(
        title=f"{selected_sku} — 30-day forecast range vs effective stock by location",
        xaxis_title="Location",
        yaxis_title="Units",
        barmode="group",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Imbalances
# ---------------------------------------------------------------------------
def render_imbalances_tab(results):
    st.header("Imbalance Flags")

    flags = results["imbalance_flags"]
    df = pd.DataFrame(flags)

    # Colour-code the flags column.
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary counts.
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    n_shortage = sum(1 for f in flags if "shortage" in f["flags"])
    n_excess = sum(1 for f in flags if "excess" in f["flags"])
    n_ageing = sum(1 for f in flags if "ageing" in f["flags"])
    n_distorted = sum(1 for f in flags if "distorted_history" in f["flags"])

    col1.metric("Shortage", n_shortage)
    col2.metric("Excess", n_excess)
    col3.metric("Ageing", n_ageing)
    col4.metric("Distorted history", n_distorted)

    # Colour-coded flag chips.
    st.subheader("Flag detail")
    colour_map = {
        "shortage": "#ffcdd2",
        "excess": "#c8e6c9",
        "ageing": "#ffe0b2",
        "distorted_history": "#e1bee7",
        "ok": "#ffffff",
    }

    chips_html = []
    for f in flags:
        flag_list = f["flags"].split(";")
        chips = "".join(
            f'<span style="background:{colour_map.get(fl, "#fff")};'
            f'padding:2px 8px;border-radius:4px;font-size:0.8em;margin-right:4px;">{fl}</span>'
            for fl in flag_list
        )
        chips_html.append(
            f'<div style="margin-bottom:4px;">'
            f'<b>{f["sku_id"]}</b> @ {f["location"]}: {chips}'
            f' — {f["reason"]}</div>'
        )
    st.markdown("".join(chips_html), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 4: Recommendations
# ---------------------------------------------------------------------------
def render_recommendations_tab(state, results):
    st.header("Transfer Recommendations")
    st.caption("All transfers are 'pending approval'. Approving or rejecting only "
               "stores the status in session state — nothing executes.")

    recs = results["recommendations"]

    if not recs:
        st.info("No recommended transfers for the current configuration.")
    else:
        st.subheader(f"Recommended ({len(recs)})")

        for i, rec in enumerate(recs):
            k = rec_key(rec)
            current_status = st.session_state["approval_status"].get(k, rec["status"])

            with st.container(border=True):
                col_info, col_btn = st.columns([4, 1])

                with col_info:
                    badge_color = {
                        "approved": "#c8e6c9",
                        "rejected": "#ffcdd2",
                        "pending approval": "#fff9c4",
                    }.get(current_status, "#fff9c4")
                    st.markdown(
                        f'<span style="background:{badge_color};padding:2px 10px;'
                        f'border-radius:4px;font-size:0.85em;font-weight:600;">'
                        f'{current_status}</span>',
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"**{rec['sku_id']}**: {rec['from_location']} → "
                        f"{rec['to_location']}  |  **{rec['quantity']} units**  |  "
                        f"Lead {rec['lead_time_days']}d  |  "
                        f"Cost \\${rec['transfer_cost_total']}  |  "
                        f"Net benefit **\\${rec['net_benefit']}**"
                    )
                    st.caption(rec["reason"].replace("$", "\\$"))

                with col_btn:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Approve", key=f"approve_{i}", type="primary"):
                            st.session_state["approval_status"][k] = "approved"
                            audit(
                                "Approval",
                                f"{rec['sku_id']} {rec['from_location']}→{rec['to_location']} "
                                f"{rec['quantity']}u — approved",
                            )
                            st.rerun()
                    with col_b:
                        if st.button("Reject", key=f"reject_{i}"):
                            st.session_state["approval_status"][k] = "rejected"
                            audit(
                                "Rejection",
                                f"{rec['sku_id']} {rec['from_location']}→{rec['to_location']} "
                                f"{rec['quantity']}u — rejected",
                            )
                            st.rerun()

    # Not recommended.
    not_recs = results["not_recommended"]
    st.subheader(f"Not recommended ({len(not_recs)})")
    if not_recs:
        nr_df = pd.DataFrame([
            {
                "SKU": r["sku_id"],
                "Route": f"{r['from_location']} → {r['to_location']}",
                "Qty": r["quantity"],
                "Lead": r["lead_time_days"],
                "Net": r["net_benefit"],
                "Reason": r["reason"],
            }
            for r in not_recs
        ])
        st.dataframe(nr_df, use_container_width=True, hide_index=True)
    else:
        st.write("(none)")

    # Replenishment needs.
    needs = results["replenishment_needs"]
    st.subheader(f"Replenishment needs ({len(needs)} lines)")
    if needs:
        total_units = sum(r["units"] for r in needs)
        st.write(f"**Total remaining shortage: {total_units:,} units**")
        rn_df = pd.DataFrame(needs).rename(columns={
            "sku": "SKU",
            "location": "Location",
            "units": "Units",
            "supplier_lead_time_days": "Lead (days)",
            "delayed_by_disruption": "Delayed",
        })
        st.dataframe(rn_df, use_container_width=True, hide_index=True)
    else:
        st.write("(none — all shortage covered by transfers)")


# ---------------------------------------------------------------------------
# Tab 5: Impact
# ---------------------------------------------------------------------------
def render_impact_tab(state, results):
    st.header("Impact: Before vs After (approved transfers only)")

    st.info(
        "Forecasts are estimates, not guarantees. The 25% unit margin is an "
        "ASSUMPTION (25% of unit_cost). The naive baseline reference below is "
        "not directly comparable (different forecast method, no safety stock)."
    )

    # Filter recommendations to only approved ones for the impact calculation.
    all_recs = results["recommendations"]
    approved_recs = []
    for rec in all_recs:
        k = rec_key(rec)
        status = st.session_state["approval_status"].get(k, rec["status"])
        if status == "approved":
            approved_recs.append(rec)

    # Recompute impact using only approved transfers.
    from engine import compute_impact, build_ageing_actions, build_replenishment_needs

    # We need to re-run compute_impact with only approved recs.
    # But compute_impact takes recommendations as input, so we can call it
    # directly with the approved subset.
    impact_rows, baseline_ref_rows, net_benefit = compute_impact(
        state, results["forecasts"], approved_recs, results["imbalance_flags"]
    )

    # Impact table.
    st.subheader("Before (smart forecast, no transfers) vs After (approved transfers applied)")
    df_impact = pd.DataFrame(impact_rows)
    st.dataframe(df_impact, use_container_width=True, hide_index=True)

    st.write(f"**Net benefit (estimated): \\${net_benefit:.2f}**")

    # Bar chart: before vs after.
    chart_metrics = ["shortage_units", "excess_units", "ageing_units"]
    before_vals = []
    after_vals = []
    for m in chart_metrics:
        row = next(r for r in impact_rows if r["metric"] == m)
        before_vals.append(float(row["before"]))
        after_vals.append(float(row["after"]))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=chart_metrics, y=before_vals,
                         marker_color="rgba(255,99,71,0.6)"))
    fig.add_trace(go.Bar(name="After", x=chart_metrics, y=after_vals,
                         marker_color="rgba(60,179,113,0.6)"))
    fig.update_layout(
        title="Before vs After (approved transfers)",
        barmode="group",
        yaxis_title="Units",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Holding cost chart.
    holding_row = next(r for r in impact_rows if r["metric"] == "holding_cost")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Before",
        x=["holding_cost"],
        y=[float(holding_row["before"])],
        marker_color="rgba(255,99,71,0.6)",
    ))
    fig2.add_trace(go.Bar(
        name="After",
        x=["holding_cost"],
        y=[float(holding_row["after"])],
        marker_color="rgba(60,179,113,0.6)",
    ))
    fig2.update_layout(
        title="Holding cost ($)",
        barmode="group",
        height=250,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Baseline reference (separate block).
    st.subheader("Naive baseline reference")
    st.caption("14-day average, no safety stock; not directly comparable.")
    st.dataframe(pd.DataFrame(baseline_ref_rows), use_container_width=True, hide_index=True)

    # --- Audit log ---
    st.subheader("Audit log")
    st.caption("All edits, toggles and approvals are logged with timestamps.")

    log = st.session_state.get("audit_log", [])
    if log:
        log_df = pd.DataFrame(log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        # Save to CSV.
        path = save_audit_log()
        st.success(f"Audit log saved to `{path}`")
    else:
        st.write("(no actions yet)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
