# Gulftrade Agent — Hackathon Data Generator

This project generates a synthetic supply-chain dataset for the Gulftrade
hackathon. All data is fictional and deterministic (fixed random seed).

## Quick start

```bash
python3 generate_data.py
```

Requirements: **Python 3.8+** with only the standard library (`csv`,
`datetime`, `random`, `os`, `math`). No third-party packages needed.

## Output

Running the script creates a `data/` folder in the project root with six
CSV files:

| File | Description |
|------|-------------|
| `skus.csv` | 20 fictional SKUs across 5 product types |
| `sales_history.csv` | 90 days of daily sales per SKU per location |
| `inventory.csv` | Current available/reserved stock, age, safety stock |
| `inbound_orders.csv` | Incoming stock with expected arrival and lead time |
| `transfer_costs.csv` | Cost per unit and lead time between location pairs |
| `disruptions.csv` | Two dated disruption events |

## Planted scenarios

The dataset intentionally contains several supply-chain problems for
participants to discover and solve. See **`answer_key.md`** for the full
list and exact parameters.

Highlights:

- **Hero case**: USB-C chargers with imbalanced stock vs. demand across
  Dubai, Abu Dhabi, and Sharjah.
- **Stockout**: A 10-day sales gap in Abu Dhabi for a specific SKU.
- **Missing records**: A few days of sales data are deliberately absent.
- **Aging stock**: Some inventory is older than 120 days.
- **Bad transfer**: One transfer route where cost exceeds the benefit.
- **Disruptions**: A component shortage (3 SKUs) and a shipping-route delay.

## Reproducibility

A fixed random seed (`RANDOM_SEED = 42`) is used so every run produces
identical output.
