# Comprehensive Exploratory Data Analysis Report

## Executive Summary
This report documents data quality, distributions, sensor relationships, and degradation dynamics across the two real industrial datasets used in the **Smart Predictive Maintenance & Failure Detection System**:
1. **AI4I 2020 Predictive Maintenance Dataset** (Milling Machine Tool Telemetry)
2. **NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset** (FD001 Run-to-Failure Records)

---

## 1. AI4I 2020 Milling Machine Analysis

### Data Hygiene & Structure
- **Total Records:** 10,000 rows
- **Missing Values:** 0 across all features (100% complete)
- **Duplicate Records:** 0
- **Machine Failures:** 339 occurrences (3.39% positive failure rate)
- **Class Imbalance Ratio:** ~28.5 : 1 (Healthy : Failed), indicating that accuracy alone is a deceptive metric. Models must be optimized for Precision, Recall, PR-AUC, and F1-score with cost-sensitive weighting.

### Failure Mechanism Breakdown
| Failure Mode Code | Description | Occurrences | Percentage of Total Failures |
| :--- | :--- | :--- | :--- |
| **HDF** | Heat Dissipation Failure | 115 | 33.9% |
| **OSF** | Overstrain Failure | 98 | 28.9% |
| **PWF** | Power Failure | 95 | 28.0% |
| **TWF** | Tool Wear Failure | 46 | 13.6% |
| **RNF** | Random Failure | 19 | 5.6% |

### Key Physical Observations & Domain Feature Opportunities
1. **Torque vs. Rotational Speed Inverse Relationship:**
   Mechanical power is given by $P = \tau \cdot \omega$. Torque and speed display an inverse hyperbolic curve. Power failures (PWF) trigger when calculated power falls below 3,500 W or exceeds 9,000 W. An explicit power proxy feature (`power_kw`) strongly isolates PWF.
2. **Thermal Dissipation:**
   Process temperature tracks ambient air temperature closely. Heat Dissipation Failures (HDF) occur when $\Delta T = T_{process} - T_{air} < 8.6\text{ K}$ alongside low rotational speed ($< 1,380\text{ rpm}$). An engineered temperature difference feature (`temp_diff_k`) serves as a direct indicator.
3. **Overstrain Product:**
   Overstrain Failure (OSF) occurs when tool wear multiplied by torque exceeds product-type dependent stress limits (11,000 min$\cdot$Nm for L, 12,000 for M, 13,000 for H). Engineering an interaction term (`tool_wear_min * torque_nm`) directly captures mechanical overstrain.

---

## 2. NASA C-MAPSS Turbofan Engine Degradation Analysis (FD001)

### Trajectory & Lifetime Statistics
- **Training Set Cycles:** 20,631 time steps across 100 engines running until catastrophic failure.
- **Test Set Cycles:** 13,096 time steps across 100 engines, truncated prior to failure for blind RUL estimation.
- **Ground Truth Target Records:** 100 true remaining useful life values for test engines.
- **Engine Lifespans:** Minimum 128 cycles, Maximum 362 cycles, Mean 206.3 cycles.

### Sensor Dynamics & Feature Pruning Insights
1. **Invariant / Flat Sensors (Zero Variance):**
   Sensors `['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']` display near-zero standard deviation ($< 10^{-5}$) across all engines and cycles under FD001 operating conditions. These sensors carry zero prognostic information and should be dropped during preprocessing to eliminate noise and computational overhead.
2. **Key Degradation Indicators (High Monotonicity with Wear):**
   - **Increasing with wear (Negative correlation with RUL):** `s_2` (LPC outlet temp), `s_3` (HPC outlet temp), `s_4` (LPT outlet temp), `s_8` (Core speed), `s_11` (HPC static pressure), `s_13` (Corrected core speed), `s_15` (Bypass ratio), `s_17` (Bleed enthalpy).
   - **Decreasing with wear (Positive correlation with RUL):** `s_7` (HPC outlet pressure), `s_12` (Fan speed), `s_20` (HPT coolant bleed), `s_21` (LPT coolant bleed).
3. **Piecewise Linear RUL Target Formulation:**
   In early cycles, an engine operates with negligible wear (normal baseline health). Wear accelerates only after initial micro-damage accumulates. Setting a capped RUL threshold ($RUL_{max} = 125$ cycles) prevents regression algorithms from learning noisy, uninformative high-RUL values during early operational stages.

---

## 3. EDA Generated Figures Catalog

1. `ai4i_failure_distribution.png`: Binary class imbalance and distribution across 5 specific failure modes.
2. `ai4i_correlation_heatmap.png`: Full correlation matrix highlighting physical interactions.
3. `ai4i_sensor_distributions.png`: Kernel density estimates of healthy vs failed machine cycles.
4. `ai4i_torque_speed_failure.png`: Phase-space envelope of torque versus rotational speed.
5. `cmapss_engine_lifetimes.png`: Histogram and distribution fit of run-to-failure cycles.
6. `cmapss_sensor_correlation_rul.png`: Sensor sensitivity and correlation ranking against engine RUL.
7. `cmapss_sensor_trends.png`: Temporal degradation trajectories of critical gas turbine sensors.
