# Testing & Demonstration

This directory contains tests and visualizations using **Indian stocks and mutual funds** data.

## Contents

| File | Description |
|------|-------------|
| `sample_data.py` | Sample monthly/daily data for Indian stocks (Reliance, TCS, Infosys, HDFC Bank, Nifty 50) and mutual funds (SBI Bluechip, Axis ELSS, Mirae Asset Large Cap, PPFAS Flexi Cap) |
| `test_functions.py` | Unit tests for all time series functions using Indian market data |
| `demo_visualization.py` | Generates plots demonstrating each analysis method |

## Running Tests

```bash
cd testing
python test_functions.py
```

## Generating Visualizations

```bash
cd testing
python demo_visualization.py
```

Plots will be saved to the `testing/plots/` directory.

## Data Sources

The sample data consists of approximate monthly closing prices and NAV values for 2023, used purely for educational and demonstration purposes:

### Stocks
- **Reliance Industries (RELIANCE)** - Monthly & Daily prices
- **Tata Consultancy Services (TCS)** - Monthly prices
- **Infosys (INFY)** - Monthly prices
- **HDFC Bank** - Monthly prices
- **Nifty 50 Index** - Monthly values

### Mutual Funds (Direct Plan NAV)
- **SBI Bluechip Fund**
- **Axis Long Term Equity (ELSS) Fund**
- **Mirae Asset Large Cap Fund**
- **PPFAS Flexi Cap Fund**
