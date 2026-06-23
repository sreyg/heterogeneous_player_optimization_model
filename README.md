# Heterogeneous Player Optimization Model
### A Game-Theoretic Analysis of Loot Box Revenue Optimization

**Hok San Ku (Jason)** · MATH 111A Game Theory · UC San Diego · December 2025  
Extended for portfolio: Monte Carlo sweep, price optimization, and real-world market validation · June 2026

---

## Overview

This project models the optimal design of loot box reward systems in video games using a **Stackelberg game framework** with heterogeneous player types and multi-tier reward structures. The original paper (MATH 111A) establishes the theory; the portfolio extension operationalizes it with a full parameter sweep and real-world validation against Steam Market data.

The central question: **what reward probabilities maximize developer revenue while maintaining sustainable player engagement?**

---

## Key Findings

### 1. Epic Tier Dominance
Medium-rarity items at ~4% drop rate generate **38% of total revenue** — more than any other single tier. This reflects an optimal accessibility-desirability tradeoff: a 28% acquisition probability at typical engagement is achievable enough to motivate spending, scarce enough to maintain prestige value.

### 2. The Pity Paradox
Guaranteed reward systems (pity mechanics) **decrease revenue by 14%** in multi-tier contexts, contrary to conventional wisdom. Only 1.2% of players reach the pity threshold, so the guarantee affects almost no one's behavior while imposing real implementation costs. Multi-tier structures already provide adequate engagement — pity systems substitute for, rather than complement, tiered reward design.

### 3. Revenue-Maximizing Price ≈ $7.75/box
A price sweep (extended from the original paper's fixed $3.86 baseline) identifies the theoretical revenue-maximizing price at ~$7.75 — roughly 2× the official CS2 key price of $2.50. The gap suggests real game developers optimize for player volume and long-term retention rather than per-transaction revenue, consistent with the paper's static-framework limitation.

### 4. Pity Creates Binary Player Exit, Not Gradual Decline
At tight pity thresholds (≥30 boxes), Casual players switch from full threshold commitment to **complete exit** as price crosses a tipping point — not a smooth quantity reduction. Whales maintain commitment at any price tested. This corner-solution behavior only emerges from crossing price × pity dimensions and is not captured in the original model.

### 5. Real-World Validation: Three Divergences
Comparing model predictions against Steam Market data and peer-reviewed loot box spending research:

| Metric | Model | Real-World Benchmark | Source |
|---|---|---|---|
| Epic tier revenue share | 38.1% | 15.8% | Steam Market, 4 cases sampled |
| Legendary tier revenue share | 27.5% | 40.1% | Steam Market (collector premium effect) |
| Whale revenue share (5% of population) | 28% | ~50% | Zendle et al. (2020), n=7,767 |
| Revenue-maximizing price | $7.75 | $2.50 (actual) | Official CS2 key price |

The Legendary divergence is explained by collector-scarcity premium in discontinued cases; the whale concentration gap suggests the model's fatigue parameters understate high-spender engagement depth.

---

## Model Architecture

### Stackelberg Framework
- **Leader**: Developer sets drop rate vector `r = (r_Common, r_Rare, r_Epic, r_Legendary)`
- **Followers**: Players observe rates, choose optimal box count `x* ≥ 0`
- **Solution concept**: Subgame perfect equilibrium via backward induction

### Player Utility Function
```
U_i(x; r, f_i) = Σ_tier [v_i,tier · (1 − (1−r_tier)^x)] − p·x − f_i·x²
```
- `v_i,tier`: player type i's valuation for tier items  
- `f_i·x²`: quadratic fatigue penalty (accelerating psychological cost)

### Player Heterogeneity (Monte Carlo)
| Type | Population | Fatigue Range | Avg Boxes |
|---|---|---|---|
| Whale | 5% | Uniform(0.03, 0.08) | 46.2 |
| Casual | 30% | Uniform(0.10, 0.20) | 13.7 |
| F2P | 65% | Uniform(0.40, 0.60) | 2.8 |

### Reward Tier Structure
| Tier | Drop Rate | Whale Value | Casual Value | F2P Value |
|---|---|---|---|---|
| Common | 50% | $30 | $15 | $8 |
| Rare | 15% | $150 | $60 | $25 |
| Epic | 4% | $400 | $180 | $60 |
| Legendary | 1% | $1,000 | $300 | $100 |

---

## Repository Structure

```
heterogeneous_player_optimization_model/
│
├── README.md
│
├── paper/
│   └── final_report.pdf              # Original MATH 111A submission (Dec 2025)
│
├── scripts/
│   ├── improved_loot_box_model.py    # Original simulation (paper baseline)
│   ├── generate_tableau_dataset_v2.py # Price × pity sweep (portfolio extension)
│   └── fetch_real_world_data.py      # Steam Market price pipeline (real-world validation)
│
└── data/
    ├── loot_box_player_panel.csv     # 108K rows: sim_id × price × pity × player_type
    ├── loot_box_tier_panel.csv       # 432K rows: + tier dimension
    ├── real_case_item_prices.csv     # Steam Market prices, 62 items across 4 cases
    ├── real_case_tier_ev.csv         # Per-tier real expected value by case
    └── real_world_validation.csv     # Model vs. real-world comparison table
```

---

## Technical Stack

| Tool | Use |
|---|---|
| Python (NumPy, SciPy, pandas) | Monte Carlo simulation, optimization (`minimize_scalar`), data pipeline |
| Steam Community Market API | Real item prices (Valve official endpoint, no auth required) |
| ByMykel/CSGO-API | Case contents and item rarity structure |
| Tableau Public | Interactive dashboard (price × pity heatmap, tier EV comparison, player segment breakdown) |
| LaTeX | Original academic paper |

---

## Data Notes and Limitations

- **Simulation data is synthetic**: fatigue parameters are sampled from calibrated uniform distributions, not observed player behavior. This is stated explicitly to avoid misrepresentation.
- **Steam Market prices reflect secondary resale value**, not primary player willingness-to-pay. The two concepts are related but not identical — this distinction matters for interpreting the tier revenue divergence.
- **All 4 sampled cases are discontinued legacy cases**: their Legendary-tier items carry collector-scarcity premium accumulated over years. An active case comparison would likely show closer alignment with the model's Epic Tier Dominance finding.
- **Within-tier item weighting assumed uniform**: Valve has not disclosed item-level weighting within a rarity tier. The community standard assumption of equal probability is used here.
- **Pity system model is static (one-shot)**: real pity systems track accumulated bad luck across sessions, creating multi-period dynamics the current model does not capture.

---

## Real-World Validation Context

Model predictions for medium-rarity optimal drop rate align with published industry rates:

| Game | Medium-Rarity Tier | Actual Drop Rate |
|---|---|---|
| This model (Epic tier) | Theoretical prediction | **4.0%** |
| CS:GO (Classified tier) | Industry practice | 3.2% |
| Genshin Impact (4-star) | Industry practice | 5.1% |

---

## References

1. King, D.L. & Delfabbro, P.H. (2014). The cognitive psychology of internet gaming disorder. *Clinical Psychology Review*.
2. Drummond, A. & Sauer, J.D. (2018). Video game loot boxes are psychologically akin to gambling. *Nature Human Behaviour*.
3. Chen, N. et al. (2021). Loot box pricing and design. *Management Science*, 67(8).
4. Zendle, D. et al. (2020). The prevalence of loot boxes in mobile and desktop games. *Addiction*, 115(9).
5. Valve Corporation (2017). CS:GO weapon case drop rate disclosure (Perfect World regulatory filing).
