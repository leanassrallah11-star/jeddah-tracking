# Jeddah Walking Trajectory — One-Step-Ahead Position Prediction

Predicts the **next GPS position** (latitude, longitude) from the **previous K positions** using Mobisense walking-mode data collected in Jeddah.

## Quick Start

```bash
pip install -r requirements.txt
python3 jeddah_trajectory_prediction.py
```

All results (metrics CSV, prediction CSVs, plots) are saved to `results/`.

## Project Structure

```
jeddah tracking/
├── walk_jeddah/                        # Raw CSV data (one file = one trajectory)
│   ├── walk_jeddah_069.csv
│   ├── walk_jeddah_070.csv
│   ├── walk_jeddah_071.csv
│   └── walk_jeddah_072.csv
├── jeddah_trajectory_prediction.py     # Main script
├── requirements.txt
├── README.md
└── results/                            # Generated outputs
    ├── metrics.csv
    ├── predictions_*.csv
    ├── pred_vs_actual_lat.png
    ├── pred_vs_actual_lon.png
    ├── error_histogram.png
    ├── error_cdf.png
    ├── trajectory_map.png
    └── lstm_loss.png
```

## Configuration

All configurable parameters are at the top of `jeddah_trajectory_prediction.py`:

| Parameter    | Default       | Description                                  |
|------------- |-------------- |--------------------------------------------- |
| `DATA_DIR`   | `"walk_jeddah"` | Folder containing CSV files                 |
| `LAT_COL`    | `"Latitude"`  | Column name for latitude in your CSVs        |
| `LON_COL`    | `"Longitude"` | Column name for longitude in your CSVs       |
| `TIME_COL`   | `"Time"`      | Column name for timestamp (set `None` if N/A)|
| `K`          | `5`           | Number of past positions as input (window)   |
| `TRAIN_RATIO`| `0.80`        | First 80% train / last 20% test              |
| `LSTM_EPOCHS`| `50`          | LSTM training epochs                         |
| `LSTM_BATCH` | `32`          | LSTM mini-batch size                         |
| `OUTPUT_DIR` | `"results"`   | Where to write outputs                       |

### How to change column names

If your CSV uses different column headers (e.g. `lat`, `lng`, `ts`), edit these three lines:

```python
LAT_COL = "lat"
LON_COL = "lng"
TIME_COL = "ts"       # or set to None if no timestamp
```

### How to change K (window size)

```python
K = 10   # use 10 past positions instead of 5
```

## Models

| Model              | Type       | Description                                     |
|------------------- |----------- |------------------------------------------------ |
| Naive Baseline     | Heuristic  | Predict next = last known position               |
| Linear Regression  | Classical  | Learns linear mapping from K past points         |
| Random Forest      | Ensemble   | 100-tree bagging on flattened K×2 features       |
| Extra Trees        | Ensemble   | 100-tree extremely randomized on K×2 features    |
| LSTM               | Deep       | PyTorch LSTM(64) → Dense(2), MinMax-normalized   |

## Evaluation Metrics

For each model, the script reports:
- **MAE** (latitude and longitude in degrees)
- **RMSE** (latitude and longitude in degrees)
- **Haversine distance** in metres — mean, median, 90th percentile

## Key Design Decisions

- **No shuffling**: time order is strictly preserved within and across trajectories.
- **No cross-file windows**: sliding windows never span two different CSV files.
- **Real positions only**: always uses actual past GPS points as input (no recursive/autoregressive prediction).
- **Consecutive deduplication**: raw CSVs have ~20× repeated rows at each GPS fix; these are collapsed to unique position changes before windowing.

## Extending with Signal Features (PCI, RSRP, etc.)

The CSV files contain many radio signal columns (RSRP, RSRQ, RSSI, CQI, PCI, etc.). To incorporate them:

1. In `load_trajectories()`, keep the signal columns alongside lat/lon when building the trajectory DataFrame (instead of extracting only `["lat", "lon"]`).
2. In `make_windows()`, increase the feature dimension from 2 to 2 + N_signals.
3. For tree models: the flattened input becomes `K × (2 + N_signals)`.
4. For LSTM: change `input_size=2` to `input_size=2 + N_signals` in `LSTMModel`.
5. Handle NaN values in signal columns (forward-fill or interpolate).

Example columns available: `RSRP/antenna port - 1..10`, `RSRQ/antenna port - 1..10`, `Physical cell identity (LTE pcell)`, `Pathloss (LTE pcell)`, `Velocity`, etc.
