<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Python][python-shield]][python-url]
[![PyMC][pymc-shield]][pymc-url]
[![License: MIT][license-shield]][license-url]

</div>

<!-- PROJECT LOGO & TITLE -->
<br />
<div align="center">
  <h1> March Madness Bayesian Simulator</h1>
  <p>
    A reusable NCAA tournament prediction engine powered by Bayesian hierarchical modeling and Monte Carlo simulation.
    <br />
    <br />
  </p>
</div>

---

## Highlights

- **Bayesian sensor fusion** — combines 5 independent rating sources (KenPom, Barttorvik, BPI, SRS, TeamRankings) into a single posterior strength estimate per team, learning each source's bias and noise
- **Fat-tailed game model** — Student-t(df=7) margin distributions produce realistic upset rates that Normal distributions miss
- **10,000 Monte Carlo simulations** — full bracket simulations with proper seed matchups, producing round-by-round advancement probabilities and championship odds
- **Interactive ESPN-style bracket** — React frontend where you click to advance teams, with Bayesian win probabilities displayed on every matchup
- **Reusable annually** — swap in new season's ratings, create a bracket JSON, and rerun. The model architecture doesn't change year to year

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>📋 Table of Contents</strong></summary>
  <ol>
    <li><a href="#-highlights">Highlights</a></li>
    <li><a href="#-architecture">Architecture</a></li>
    <li><a href="#-model-specification">Model Specification</a></li>
    <li><a href="#-data-sources">Data Sources</a></li>
    <li>
      <a href="#-quick-start">Quick Start</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#-usage">Usage</a></li>
    <li><a href="#-2026-predictions">2026 Predictions</a></li>
    <li><a href="#-learned-parameters">Learned Parameters</a></li>
    <li><a href="#-interactive-bracket">Interactive Bracket</a></li>
    <li><a href="#-project-structure">Project Structure</a></li>
    <li><a href="#-adding-a-new-year">Adding a New Year</a></li>
    <li><a href="#-name-mapping-system">Name Mapping System</a></li>
    <li><a href="#-roadmap">Roadmap</a></li>
    <li><a href="#-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

## Architecture

Three-layer design where each component is independently swappable:

```
  ┌───────────────────────────────────────────────────────────────┐
  │                    LAYER 1: TEAM STRENGTHS                    │
  │                                                               │
  │  KenPom ─┐                                                   │
  │  Barttor ─┤  Observation     Seed         Posterior           │
  │  BPI ─────┼─→  Model    ──→ Hierarchy ──→  θ per team        │
  │  SRS ─────┤  (a_k, b_k,     (α, β,       (Student-t          │
  │  TeamRnk ─┘   σ_obs_k)      σ_seed)       thick tails)       │
  ├───────────────────────────────────────────────────────────────┤
  │                    LAYER 2: GAME SIMULATOR                    │
  │                                                               │
  │  margin ~ Student-t(df=7, θ_A - θ_B, σ_game)                 │
  │  P(A wins) = P(margin > 0)                                   │
  ├───────────────────────────────────────────────────────────────┤
  │                  LAYER 3: BRACKET SIMULATOR                   │
  │                                                               │
  │  For each of 10,000 simulations:                              │
  │    1. Draw (θ, σ_game) from posterior                         │
  │    2. Simulate all 63 games using bracket structure           │
  │    3. Record advancement for every team at every round        │
  └───────────────────────────────────────────────────────────────┘
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Model Specification

### Bayesian Hierarchical Measurement Error Model

**Level 1 — Seed priors** place each team's expected strength based on their tournament seed:

```
α ~ Normal(0, 50)           β ~ Normal(-2, 2)
σ_seed ~ Half-Normal(5)
μ_seed[s] ~ Normal(α + β·s, σ_seed)     for s = 1..16
```

**Level 2 — Team strengths** allow individual teams to deviate from their seed expectation:

```
σ_team ~ Half-Normal(5)
θ_i ~ Student-t(ν=7, μ_seed[s_i], σ_team)
```

**Level 3 — Observation model** learns each source's calibration (bias `a`, scale `b`, noise `σ`):

```
Source 0 = anchor: a=0, b=1
σ_obs[k] ~ Half-Normal(5)
b[k] = exp(Normal(0, 0.3))    a[k] ~ Normal(0, 10)     for k > 0
y_i^(k) ~ Normal(a[k] + b[k]·θ_i, σ_obs[k])
```

**Level 4 — Game outcomes** inform strength estimates from 2008–2019 tournament margins:

```
σ_game ~ Half-Normal(12)
margin_ij ~ Student-t(ν=7, θ_i - θ_j, σ_game)
```

### Why this model?

| Decision | Rationale |
|----------|-----------|
| **Not BTD** | We have pre-computed ratings, not raw game results → measurement error model, not paired comparison |
| **Student-t(df=7)** | Fat tails for realistic upset rates. Normal underestimates 12-over-5 upsets by ~40% |
| **Raw AdjEM scale** | Keeps θ and game margins on the same scale. Z-scoring broke this relationship |
| **5-source fusion** | More sources → tighter posteriors. The model learns that TeamRankings (σ=3.3) is 3× noisier than KenPom (σ=1.1) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Data Sources

| Source | Role | Coverage | Noise (σ_obs) |
|--------|------|----------|:-------------:|
| **KenPom** AdjEM | Anchor (composited w/ Barttorvik) | 2001–2026 | 1.09 |
| **Barttorvik** AdjEM | Composited → Source 1 | 2008–2026 | — |
| **TeamRankings** Predictive | Source 2 | 2008–2026 | 3.34 |
| **ESPN BPI** | Source 3 | 2008–2026 | 1.34 |
| **SRS** (Sports Reference) | Source 4 | 2008–2026 | 1.10 |
| **Big Dance** results | Game margins for Layer 4 | 2008–2019 | — |

> **Source 1** = mean(KenPom, Barttorvik) — composited to avoid correlated residuals between two AdjEM-family systems.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- ~4 GB RAM for MCMC sampling

### Installation

```bash
git clone https://github.com/yourusername/bracket-predictor.git
cd bracket-predictor

pip install pymc arviz numpy pandas scipy matplotlib
```

### Run the full pipeline

```bash
# 1. Preprocess ratings from all sources
cd etl
python preprocess.py

# 2. Fit the Bayesian model (~60 min for full run)
cd ../engine
python fit.py \
  --data-dir ../etl/data/processed \
  --output-dir output \
  --chains 4 --draws 2000 --tune 2000

# 3. Simulate the tournament
python simulate.py \
  --year 2026 \
  --trace output/trace.nc \
  --data-dir ../etl/data/processed \
  --n-sims 10000
```

<details>
  <summary><strong>Quick test run (~5 min)</strong></summary>

  ```bash
  python fit.py \
    --data-dir ../etl/data/processed \
    --output-dir output \
    --chains 2 --draws 500 --tune 500
  ```

  Convergence targets: R-hat < 1.01, ESS > 1000.
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Usage

### Tournament simulation (advancement probabilities)

```bash
python simulate.py --year 2026 --trace output/trace.nc \
  --data-dir ../etl/data/processed --n-sims 10000
```

### Game-by-game predictions with spreads

```bash
# First round matchups
python predict_games.py --year 2026 --trace output/trace.nc \
  --data-dir ../etl/data/processed

# Full bracket (advance favorites through every round)
python predict_games.py --year 2026 --trace output/trace.nc \
  --data-dir ../etl/data/processed --full
```

### Leave-one-year-out backtesting

```bash
python diagnostics.py \
  --data-dir ../etl/data/processed \
  --output-dir output \
  --chains 2 --draws 500 --tune 500
```

Produces calibration plots, expected calibration error (ECE), accuracy, and log scores across 12 held-out tournament years (2008–2019).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 2026 Predictions

*10,000 Monte Carlo simulations from posterior predictive distribution*

| Team | Seed | Region | Champ % | Final Four % | Elite 8 % | Sweet 16 % |
|------|:----:|--------|--------:|------------:|---------:|-----------:|
| **Duke** | 1 | East | **25.8** | 39.5 | 64.2 | 78.5 |
| **Michigan** | 1 | Midwest | **22.8** | 35.3 | 57.3 | 75.3 |
| **Arizona** | 1 | West | **17.3** | 33.6 | 52.6 | 73.8 |
| Florida | 1 | South | 8.1 | 19.0 | 35.9 | 62.8 |
| Houston | 2 | South | 4.9 | 12.5 | 26.0 | 47.3 |
| Iowa State | 2 | Midwest | 4.7 | 9.9 | 23.0 | 57.2 |
| Illinois | 3 | South | 4.6 | 11.5 | 23.9 | 44.9 |
| Purdue | 2 | West | 4.3 | 11.6 | 24.1 | 54.6 |
| UConn | 2 | East | 1.7 | 3.8 | 11.4 | 38.7 |
| Gonzaga | 3 | West | 1.1 | 4.5 | 11.7 | 33.1 |

> Duke, Michigan, and Arizona combine for a **65.9%** championship probability — driven by AdjEM ratings (38.9, 37.6, 37.7) that are 5+ points clear of the field.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Learned Parameters

*Posterior means from 4 chains × 2,000 draws (R-hat < 1.004, ESS > 1,100)*

| Parameter | Value | What it means |
|-----------|------:|---------------|
| α | 30.0 | Hypothetical 0-seed AdjEM |
| β | -1.71 | AdjEM drop per seed line (~1.7 pts) |
| σ_seed | 2.77 | How much seeds vary from the linear trend |
| σ_team | 2.74 | How much a team can deviate from its seed's expectation |
| σ_game | 10.54 | Single-game noise — a 10-point favorite wins ~81% |
| σ_obs[0] | 1.09 | KenPom/Barttorvik composite noise (most precise) |
| σ_obs[1] | 3.34 | TeamRankings noise (3× noisier than KenPom) |
| σ_obs[2] | 1.34 | BPI noise |
| σ_obs[3] | 1.10 | SRS noise (nearly as precise as KenPom) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Project Structure

```
bracket-predictor/
├── etl/
│   ├── preprocess.py              # Main ETL: load, merge, validate all sources
│   ├── standardize_names.py       # KENPOM_TO_CANONICAL master name mapping
│   ├── scrapers.py                # Selenium/requests scrapers for BPI, SRS, TR
│   └── data/
│       ├── raw/                   # Per-source CSVs (kenpom/, barttorvik/, bpi/, srs/, tr/)
│       └── processed/
│           ├── ratings_matrix_standardized.csv   # Model input: all teams × all sources
│           ├── tournament_games.csv              # Game margins (2008–2019)
│           └── bracket_2026.json                 # Region assignments
│
├── engine/
│   ├── model.py                   # PyMC model definition + prediction model
│   ├── fit.py                     # MCMC fitting (4 chains, NUTS sampler)
│   ├── simulate.py                # Monte Carlo bracket simulation
│   ├── predict_games.py           # Game-by-game matchup predictions + spreads
│   ├── diagnostics.py             # Leave-one-year-out CV + calibration
│   └── output/
│       ├── trace.nc               # Posterior samples (ArviZ InferenceData)
│       └── simulation_2026.csv    # Results
│
└── frontend/
    └── bracket_espn.jsx           # Interactive React bracket
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Adding a New Year

Each March, run these steps to generate predictions for the new tournament:

1. **Download ratings** — get KenPom pre-tournament AdjEM for all tournament teams; scrape BPI, SRS, TeamRankings, and Barttorvik for the new season
2. **Create bracket JSON** — save `data/processed/bracket_{year}.json`:
   ```json
   {
     "regions": {
       "East": ["Duke", "Siena", ...],
       "South": ["Florida", ...],
       "West": ["Arizona", ...],
       "Midwest": ["Michigan", ...]
     }
   }
   ```
3. **Fix seeds** — if your KenPom data has incorrect seeds, add overrides in `load_kenpom()`
4. **Run pipeline** — `preprocess.py` → `fit.py` → `simulate.py`

> The model hyperparameters (α, β, σ_seed, etc.) are shared across all years, so adding a new year doesn't require re-architecting anything.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Name Mapping System

NCAA team names vary wildly across sources. A two-step chain resolves this:

```
Source raw name  →  KenPom-style name  →  Canonical name
   "Iowa St"    →     "Iowa St."      →   "Iowa State"
   "ISU" (BPI)  →     "Iowa St."      →   "Iowa State"
   "Iowa State" →     "Iowa St."      →   "Iowa State"
    (SRS)
```

Four dictionaries in `preprocess.py`:
- `BPI_ABBREV_TO_KENPOM` — ESPN all-caps abbreviations (341 teams mapped)
- `SRS_TO_KENPOM` — Sports Reference full names
- `TR_SHORT_TO_KENPOM` — TeamRankings short names
- `KENPOM_TO_CANONICAL` — KenPom → final canonical form

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Roadmap

- [x] Bayesian hierarchical measurement error model
- [x] 5-source sensor fusion (KenPom, Barttorvik, BPI, SRS, TeamRankings)
- [x] Monte Carlo bracket simulation (10,000 runs)
- [x] Game-by-game predictions with spreads
- [x] Interactive ESPN-style React bracket
- [x] 2026 tournament predictions
- [ ] Leave-one-year-out backtesting results + calibration plots
- [ ] 2021–2025 game results for expanded backtesting
- [ ] Play-in game simulation (currently resolved by AdjEM)
- [ ] Tempo-adjusted four factors as model covariates
- [ ] Live updating during tournament (feed in actual results)
- [ ] Historical accuracy comparison vs Vegas lines

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Acknowledgments

- [KenPom](https://kenpom.com/) — the gold standard for college basketball efficiency ratings
- [Barttorvik](https://barttorvik.com/) — independent AdjEM ratings with excellent historical data
- [PyMC](https://www.pymc.io/) — probabilistic programming framework powering the model
- [ArviZ](https://arviz-devs.github.io/arviz/) — Bayesian model diagnostics and visualization
- [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template) — README template inspiration

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- MARKDOWN LINKS & IMAGES -->
[python-shield]: https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[pymc-shield]: https://img.shields.io/badge/PyMC-5.0+-003366?style=for-the-badge
[pymc-url]: https://www.pymc.io/
[license-shield]: https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge
[license-url]: LICENSE.txt
