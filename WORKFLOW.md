# DigitalTwin.ai — Complete Workflow Guide

> **Read this first.** This explains every step of the pipeline in plain English —
> what runs, why, what it produces, and how everything connects.

---

## The Core Idea

We have **no real factory data**. So we build a *simulator* that pretends to be
an 18-station car assembly line. The simulator generates fake-but-realistic data
including sensor readings, cycle times, and deliberately injected faults.

We then train ML models on that fake data, and ask: *can the model detect a fault
before a human would notice it?* The answer is yes — and that is the demo.

---

## The Assembly Line (what we're simulating)

Imagine 18 workstations arranged in a single chain:

```
[S0] → [S1] → [S2] → ... → [S17]
```

A car body moves through each station in order. Each station does one job
(welding, painting, brake install, etc.) and passes the car to the next.

**Three types of stations:**

| Type | What data they produce | How common |
|---|---|---|
| Well-instrumented | Temperature + Vibration + Torque sensors + proxy signals | 72% (13 stations) |
| Proxy-only | Only cheap proxy signals (motor current, setpoint error) | 17% (3 stations) |
| Manual | Only a timestamp + pass/fail tick | 11% (2 stations) |

---

## The Two Fault Types

We inject two kinds of faults at controlled times:

1. **Bottleneck** — A station slows down. Its cycle time inflates (e.g. S3 takes
   2× as long). This causes a *queue* to build up at S4, S5, etc. downstream.
   The ripple is real — if S3 is slow, S4 has to wait, so its effective cycle
   time also increases. This is the key signal the GCN model learns to anticipate.

2. **Defect** — A station silently starts producing bad outputs. The *quality*
   drifts in a repeating sine wave, but the cycle time looks nearly normal.
   This is harder to catch because there's no obvious spike.

---

## Step-by-Step Pipeline

### STEP 1 — Simulator generates data
**File:** `simulator/assembly_line.py`

Run: `python eval/validate_stage1.py`

What happens:
- The simulator runs for N minutes (one row per station per minute)
- For each station, it computes:
  - True sensor values (temp, vibration, torque) with realistic drift + noise
  - Proxy signals — a *nonlinear* function of the true sensors + extra noise
    (so proxy ≠ true, which makes learning meaningful)
  - Queue depth: how much upstream backlog has accumulated
  - Cycle time: baseline + noise + bottleneck factor + queue wait
- Faults are injected at scripted times and logged separately
- **The fault log is NEVER given to the model** — it's used only to score detections later

Output files saved to `data/runs/`:
```
run_0000_obs.parquet    ← what the model sees (NaN for hidden sensors)
run_0000_gt.parquet     ← hidden ground truth (used only at eval time)
run_0000_faults.csv     ← fault labels (used only for scoring)
```

---

### STEP 2 — Soft Sensor fills in proxy-only stations
**File:** `soft_sensor/model.py`

Run: `python eval/validate_stage2.py`

**The problem:** Proxy-only stations have no temperature/vibration/torque readings.
But the model needs those values.

**The solution:** Train a GradientBoosting model to *predict* true sensor values
from proxy signals, using well-instrumented stations as training data.

```
Training data:  well-instrumented stations
                (we have both proxy signals AND true sensors here)

Input features: motor_current, setpoint_error, cycle_duration_s
                + their lags (t-1, t-2, t-3)
                + cycle_time, queue_depth, baseline_cycle

Output:         estimated temperature, estimated vibration, estimated torque
```

After training, we apply this model to proxy-only stations to generate
`est_temperature`, `est_vibration`, `est_torque` columns — estimates that
replace the missing real readings.

Manual stations (no proxy signals at all) are left as NaN here. They are
handled by the GCN's neighbour-propagation in Step 4.

---

### STEP 3 — Build the Station Graph
**File:** `model/graph.py`, `configs/graph.yaml`

Run: `python eval/validate_stage3.py`

This is simple: we define the graph as:
- **Nodes** = stations 0 through 17
- **Edges** = directed S0→S1, S1→S2, ..., S16→S17

This graph represents physical flow. The GCN will use it to pass information
from upstream to downstream stations.

---

### STEP 4 — Train the Core LSTM + GCN Model  ⚠️ LONGEST STEP (~3 min)
**File:** `model/network.py`, trained in `eval/validate_stage4.py`

Run: `python eval/validate_stage4.py`

**Why two models glued together?**

- **LSTM** (Long Short-Term Memory): understands *time patterns* within a single station.
  Each station has its own LSTM that reads its last 15 minutes of history and
  produces a hidden state vector summarising "what's going on at this station right now."

- **GCN** (Graph Convolutional Network): understands *spatial patterns* across stations.
  After all LSTMs run, the GCN passes each station's hidden state to its downstream
  neighbour. So S4's representation gets updated with information from S3's state.
  This is what lets the model anticipate a ripple BEFORE it arrives at S5.

**Two output heads:**

| Head | What it predicts | Used for |
|---|---|---|
| Short-horizon | Cycle time for next 5 minutes (point estimate) | Anomaly detection, Floor Supervisor |
| Long-horizon | Cycle time for next 30 minutes (mean + confidence band) | Queue forecast, Plant Manager |

**Training data:**
- 20 simulation runs × 2 hours each, with random fault injections
- Validation: 5 separate runs with different seeds (never seen in training)

**Ablation result (key evidence):**

| Model variant | Forecast error at S5 during S3 bottleneck |
|---|---|
| LSTM only (no graph) | 4.58 s MAE |
| LSTM + GCN | **3.82 s MAE** (16% better) |

The GCN wins because it sees S3's state and anticipates the wave hitting S5.
An LSTM-only model must wait until S5's own readings start rising.

**Saves:** `data/model/core_model.pt`

---

### STEP 5 — Anomaly Detection + Attribution
**File:** `anomaly/detector.py`

Run: `python eval/validate_stage5.py`

**Step 1: Compute residual**
```
residual = |actual_cycle_time - predicted_cycle_time|
```

**Step 2: Compute dynamic threshold**
```
threshold = mean(recent_normal_residuals) + 3 × std(recent_normal_residuals)
```
The threshold adapts to the station's normal noise level. During an anomaly,
it **stops updating** (otherwise the threshold would adapt to the fault and miss it).

**Step 3: Flag if residual exceeds threshold**

**Step 4: Classify the anomaly type** (rule scoring across 3 classes):
- `bottleneck` — cycle time is *higher* than predicted + sustained + rising trend
- `defect` — sustained but *oscillating* residuals (sine wave pattern), moderate size
- `sensor-fault` — isolated spike, not sustained, no upstream correlation

**Step 5: Attribute to origin station**
- *Timing check*: is the upstream station (i-1) currently also anomalous?
- *Weight check*: is the upstream residual big relative to this station's residual?
- If both → "likely propagated from S{i-1}" with high confidence
- If neither → "local fault at this station"

---

### STEP 6 — Prescriptive Actions
**File:** `action/engine.py`, rules in `configs/rules.yaml`

After anomaly detection, a simple rule engine reads the anomaly state and
produces actionable text recommendations:

| Situation | Rule | Recommendation |
|---|---|---|
| Queue forecast ratio > 1.5× | QUEUE_BUILDUP_WARNING | Reduce input rate at feeder station |
| Queue forecast ratio > 2.0× | QUEUE_BUILDUP_CRITICAL | Supervisor review required |
| Anomaly with attribution confidence > 50% | ANOMALY_WITH_ATTRIBUTION | Inspect **origin** station (not every downstream one) |
| Defect pattern detected | DEFECT_DETECTED | Halt output for quality check |

The key insight: attribution collapses "10 downstream stations are flashing red"
into **one recommendation**: "inspect S3".

---

### STEP 7 — Dashboard
**File:** `dashboard/app.py`

Run: `streamlit run dashboard/app.py`

Three tabs, one data backend:

**Tab 1 — Floor Supervisor**
- Live station topology (nodes coloured red/orange/yellow/blue by anomaly type)
- Select any station → see cycle time history + GCN forecast (5-min point + 30-min band)
- Attribution banner: "Anomaly at S4 likely caused by S3 (80% confidence)"
- Residual gauge: live needle vs threshold

**Tab 2 — Plant Manager**
- Bar chart: which stations flagged most often this shift?
- Line chart: % of stations anomalous over time (with fault windows highlighted)
- KPI tiles: disrupted minutes, detection lead time, defect exposure

**Tab 3 — Leadership**
- 4 large numbers: disruption caught, lead time, defective units avoided, protected output
- Plain-English executive narrative
- How-it-works explanation for non-technical audience

---

## Quick Start (after all libraries installed)

```bash
# ONE-TIME: train models (takes ~3 min)
python eval/validate_stage4.py

# EVERY TIME: run the dashboard
streamlit run dashboard/app.py
```

The dashboard opens at **http://localhost:8501**

- Slider at top-left controls which minute of the simulation you're looking at
- Scrub to **t=75** to see the bottleneck fault (injected at S3, t=60–120)
- Scrub to **t=160** to see the defect fault (injected at S8, t=150+)
- Watch the line topology turn red, the attribution banner appear, and the
  actions panel in the sidebar activate

---

## File Map

```
simulator/assembly_line.py   ← generates all synthetic data
simulator/line_config.py     ← loads configs/line_config.yaml

soft_sensor/model.py         ← GradientBoosting proxy→sensor estimator

model/network.py             ← LSTM + GCN architecture
model/dataset.py             ← rolling-window dataset builder
model/graph.py               ← loads configs/graph.yaml edge list

anomaly/detector.py          ← residual threshold + type classification + attribution

action/engine.py             ← evaluates configs/rules.yaml if/then rules
configs/rules.yaml           ← all 4 prescriptive rules (human-readable)

dashboard/api.py             ← connects all layers into one API object
dashboard/app.py             ← Streamlit 3-tab UI

eval/validate_stage*.py      ← one validation script per stage (run in order)
run_demo.py                  ← single script that runs the entire pipeline
```
