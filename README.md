# Bike Rental Monitoring with Evidently

Monitoring of a bike rental model (UCI Bike Sharing dataset, hourly data) with **Evidently 0.7.2**.
Reference = January 2011, production = February 2011, split into weeks W1, W2, W3.

Repository: https://github.com/christophneddens/examen_evidently

## Project structure

```
.
├── src/main.py                # complete script
├── data/                      # downloaded data (hour.csv)
├── bike_sharing_monitoring/   # Evidently workspace, project "Bike Rental Monitoring"
├── diagnostic_report.html     # drift diagnosis of the worst week
├── Dockerfile
├── docker-compose.yml         # services monitor-script and monitor-ui
├── requirements.txt
└── README.md
```

## How to run

```bash
docker compose up --build
```

1. `monitor-script` runs `src/main.py`: download, training, validation, one snapshot per week, drift diagnosis.
2. When the script has finished successfully, `monitor-ui` starts the Evidently UI on port 8000.
3. Open http://localhost:8000 → project **Bike Rental Monitoring** (5 reports: validation, W1, W2, W3, diagnosis).
   On a remote VM, first open a tunnel: `ssh -i <key.pem> -L 8000:localhost:8000 ubuntu@<vm-ip>`.

Stop with `Ctrl+C`, then `docker compose down`.

## What the script does (src/main.py)

| Step | Function | Content |
|---|---|---|
| 1 | `download_data` | downloads and extracts the zip file (only once) |
| 2 | `load_data` | January 2011 = reference (688 rows), February 2011 = production (649 rows) |
| 3 | `train_model`, `validate_model` | 70/30 split of January, `RandomForestRegressor`; Report with tests `RMSE < 50` and `R2 > 0.7`, read with `snapshot.dict()` |
| 4 | `create_project` | Workspace `bike_sharing_monitoring`, project "Bike Rental Monitoring" (recreated on each run) |
| 5 | `monitor_weeks` | one `RegressionPreset` snapshot per week (W1 = 1–7, W2 = 8–14, W3 = 15–21 Feb 2011) |
| 6 | `diagnose` | worst week: `ValueDrift(column="cnt")` (target drift) + `DataDriftPreset` on the numerical features, saved as snapshot and as `diagnostic_report.html` |

Choices:
- Features: `temp`, `atemp`, `hum`, `windspeed`, `hr`, `weekday`, `season`, `holiday`, `workingday`, `weathersit`. `casual` and `registered` are left out because `cnt = casual + registered` (leakage).
- Reference for the weekly reports and the diagnosis = the 30 % test part of January (data the model did not train on).
- Technical validation result: RMSE 15.6 < 50 and R² 0.90 > 0.7 → model validated for production.
- Docker image `python:3.12-slim`, because scikit-learn 1.8.0 needs Python ≥ 3.11.
- `evidently ui --host 0.0.0.0`, otherwise the UI is not reachable from outside the container.
- Each week has fewer than 1,000 rows, so Evidently uses the K-S test: drift if p-value < 0.05.

## Analysis

### 1. Performance analysis — which week shows the most significant drop?

| | January (test part) | W1 | W2 | **W3** |
|---|---|---|---|---|
| RMSE | 15.6 | 22.4 | 22.9 | **39.3** |

**W3 (15–21 February)** shows the most significant drop: its RMSE is almost twice that of W1 and W2 and 2.5 times the January value.
In the W3 report, the MAE is 25.6 against 10.9 on the reference, and the **mean error is −22.3**
(Evidently computes prediction − actual): the model **systematically predicts too few rentals**,
about 22 bikes per hour. The graph *Predicted vs Actual in Time* shows it: from 15 to 18 February
the actual rentals rise to about 120 per hour (daily average) while the predictions stay around 70.

### 2. Drift analysis — which factor explains the drop?

Diagnosis of W3 against January (`diagnostic_report.html`):

- **Target drift:** detected on `cnt` (p-value ≈ 0.00). Average rentals per hour: 55.5 → 84.2 (+52 %); highest value: 249 → 327.
- **Data drift:** detected on `temp`, `atemp`, `hum` and `windspeed`; **no** drift on `hr` and `weekday`. 4 of 6 numerical columns drifted, so the test "Share of Drifted Columns < 0.5" failed.

| Weather (average) | January | W3 | in real units (data/Readme.txt) |
|---|---|---|---|
| temp (× 41) | 0.197 | 0.363 | ≈ 8 °C → ≈ 15 °C |
| atemp, felt (× 50) | 0.204 | 0.353 | ≈ 10 °C → ≈ 18 °C |
| hum (× 100) | 0.574 | 0.424 | 57 % → 42 % |
| windspeed (× 67) | 0.197 | 0.297 | 13 → 20 (+50 %) |

**Explanation:** the drop is explained by the **weather**. W3 was much warmer (about +7 °C), drier and windier than
January, and warm weather brings more people onto bikes, hence the target drift. The calendar features (`hr`, `weekday`)
did not drift, so the cause is not the timing. The model was trained on January only and has seen hardly any such mild
hours (data drift towards the edge of the training range). A random forest also cannot predict more than the values it saw in
training (at most 249 rentals per hour in January), so it under-predicts the busy hours of W3.

### 3. Remediation — which automatic action in production?

**Automatic retraining**, triggered by the monitoring:

- **Trigger:** the weekly report shows drift on the target or on the main weather features (`temp`, `atemp`)
  **and** the RMSE rises clearly above the reference (for example more than 1.5 × the January value).
- **Action:** retrain on recent data (sliding window that includes the new weeks); rentals are counted every hour, so new labels are available quickly.
  Before deployment, the new model must pass the same tests (RMSE < 50, R² > 0.7).
- **Long term:** train on at least a full year, so that all seasons and temperatures are covered.

The other options:
- A **weather alert** is useful as an early warning (drift on `temp` is visible before the errors grow), but it does not fix the model. It is a notification, not the main action.
- A **service shutdown** is not justified. The model still follows the daily pattern (hours, weekdays) and its errors are larger but not absurd; stopping it would leave the service without any prediction. A shutdown fits data-quality failures (missing or broken data), not seasonal drift.
