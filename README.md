# DigitalTwin.ai — Predictive-Prescriptive Assembly Line Digital Twin

> **Accenture Innovation Challenge — Round 2 Hackathon Prototype**

A proof-of-concept digital twin for a vehicle assembly line that predicts
operational anomalies and prescribes corrective actions before they cause
disruptions.

---

## Problem

Vehicle assembly lines are complex, multi-station systems where a slowdown
or defect at one station propagates invisibly downstream — often only
surfacing as scrap, rework, or downtime much later. Existing monitoring
systems alert *after* the fact. This prototype demonstrates a model that
forecasts station state 5–30 minutes ahead and flags anomalies with attributed
origin stations, enabling earlier intervention.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Simulator (Stage 1)                                               │
│  18-station sequential line → per-station parquet                  │
│       true sensors + proxy signals + queue state + fault labels    │
└────────────────────┬───────────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Soft Sensor (S2)   │  fills proxy-only stations
          │  GradientBoosting   │  with estimated sensor values
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Graph (Stage 3)    │  station_i → station_{i+1}
          └──────────┬──────────┘
                     │
     ┌───────────────▼──────────────────┐
     │  LSTM + GCN Model (Stage 4)      │
     │  per-station LSTM encoder        │
     │  → GCN (propagates upstream     │
     │    state to downstream nodes)    │
     │  → short-horizon head (5 steps) │
     │  → long-horizon head (30 steps) │
     └───────────────┬──────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Anomaly (Stage 5)  │  residuals → threshold → type
          │  + Attribution      │  → likely origin station
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Actions (Stage 6)  │  if/then rules → recommendations
          └──────────┬──────────┘
                     │
     ┌───────────────▼──────────────────┐
     │  Dashboard (Stage 7)             │
     │  Tab 1: Floor Supervisor         │
     │  Tab 2: Plant Manager            │
     │  Tab 3: Leadership               │
     └──────────────────────────────────┘
```

---

## Why No Real Data?

We have no access to a real factory. The **simulator is the ground truth**
— it generates physically-plausible data with realistic noise, sensor
coverage gaps, and scriptable fault injection. The soft-sensor and
predictive models are trained on this synthetic data, and the simulator's
fault log (never given to any model as input) is used to score detection
performance. This approach is intentional and common in early-stage
industrial AI prototyping.

---

## Station Configuration

| Type | Count | % | What they output |
|---|---|---|---|
| Well-instrumented | 13 | 72% | True sensor values (temp/vibration/torque) + proxy signals |
| Proxy-only | 3 | 17% | Proxy signals only (motor current, setpoint error, cycle time) |
| Manual | 2 | 11% | Cycle start/stop timestamps + pass/fail tick only |

> A real assembly line would have 30–50 stations. The approach generalises
> directly — add more nodes to the graph and the GCN handles the rest.

---

## Train / Validation Split

| Split | Runs | Seed range | Fault placements |
|---|---|---|---|
| Training | 200 | 0 – 199 | Random, uniform across station × fault-type space |
| Validation | 40 | 1000 – 1039 | Different seed range; includes fault placements never seen in training |

The validation set acts as a proxy for "out-of-distribution generalisation"
since we cannot validate against a real factory.

---

## Fault Types Modelled

1. **Bottleneck** — a station's cycle time inflates for a sustained window.
   The ripple propagates as queue buildup downstream.
2. **Defect** — a station silently drifts into a bad output pattern that
   repeats until fixed, without an obvious sensor spike.

> **Future work** could add: sensor calibration drift, mechanical wear
> curves, multi-line topologies, RL-based action layer.

---

## Ablation Result (Stage 4)

LSTM-only vs. LSTM+GCN forecast MAE at the station **two hops downstream (S5)**
during a bottleneck injected at S3 (5-minute ahead forecast):

| Model | Station N+2 (S5) MAE during bottleneck |
|---|---|
| LSTM only | **6.23 s** |
| LSTM + GCN | **4.78 s** (23% better) |

The GCN layer allows S5's representation to be informed by S3's hidden state,
so the model anticipates the queue wave before it physically arrives.
An LSTM-only model has no cross-station signal and must wait until the wave
reaches S5's own input sequence.

---

## Detection Metrics (Stage 5)

Results from `eval/validate_stage5.py` on a held-out 4-hour run (seed=404):

| Metric | Value |
|---|---|
| Bottleneck detected (S3, t=60–100) | **Yes** — 2 flags in fault window |
| Defect detected (S8, t=140+) | **Yes** — 2 flags in fault window |
| Stage 7 API: S3 residual at t=90 | **16.70 s** (well above 3σ threshold) |
| Stage 7 API: actions triggered at t=90 | **4 prescriptive actions** |
| Anomaly type classification | bottleneck / defect / sensor-fault via rule scoring |
| Attribution | upstream timing + edge-weight proxy; origin station identified |

*Full precision/recall sweep available in `eval/validate_stage5.py` output.*

---

## How to Run

```bash
# 1. Install
pip install -e .

# 2. Stage 0 — scaffold check
python eval/validate_stage0.py

# 3. Stage 1 — simulator validation + plots
python eval/validate_stage1.py

# 4. Stage 2 — soft sensor train + evaluate
python eval/validate_stage2.py

# 5. Stage 3 — graph construction check
python eval/validate_stage3.py

# 6. Stage 4 — train LSTM+GCN model + ablation (saves core_model.pt)
python eval/validate_stage4.py

# 7. Stage 5 — anomaly detection on held-out run
python eval/validate_stage5.py

# 8. Stage 6 — action engine rule validation
python eval/validate_stage6.py

# 9. Stage 7 — dashboard API integration test
python eval/validate_stage7.py

# 10. Full end-to-end demo (requires core_model.pt from Stage 4)
python run_demo.py

# 11. Launch three-tier dashboard
streamlit run dashboard/app.py
```

---

## Assumptions

- Timestep granularity: 1 minute
- Baseline cycle times: 55 – 120 seconds (see `configs/line_config.yaml`)
- Sensor noise: ±3% of baseline (true sensors), ±5% (proxy signals)
- Queue ripple EMA alpha: 0.3
- Short-horizon forecast window: 5 timesteps (5 min)
- Long-horizon forecast window: 30 timesteps (30 min)
- Anomaly threshold: mean + 3σ of normal-operation residuals

---

## Future Work

- RL-based action layer (replace if/then rules with a policy)
- Real hospital-/factory-scale line (30–50 stations)
- Additional fault types: sensor calibration drift, wear curves
- Edge weights learned end-to-end via attention GCN
- Deployment on edge hardware near the line controller
