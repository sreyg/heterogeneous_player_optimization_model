"""
TABLEAU DATASET GENERATOR v2 — Price x Pity Sweep
====================================================
Extends generate_tableau_dataset.py per paper Section 3.7
("Dynamic pricing and box types" extension): instead of holding
PRICE_PER_BOX fixed at $3.86, sweep across a price range and cross
it with the existing pity-threshold sweep.

This produces a genuine 2-variable grid (Price x Pity Threshold)
suitable for a heatmap, rather than a single-variable line chart.

Output:
  loot_box_player_panel.csv  (grain: sim_id x price x pity x player_type)
  loot_box_tier_panel.csv    (grain: sim_id x price x pity x player_type x tier)

NOTE: filenames match the original files on purpose so the Tableau
relationship structure carries over. IMPORTANT: the join key now
needs a 4th field — price_per_box — alongside sim_id, pity_threshold,
player_type. See instructions after running.

Author: Jason (extension of MATH 111A project for portfolio)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# ============================================================================
# PARAMETERS
# ============================================================================

BASELINE_PRICE = 3.86  # CS:GO-calibrated price from the paper
# Revenue peaks near $7.75 (confirmed via deterministic pre-check) -- grid below
# brackets baseline, approaches the peak, hits it, then shows the decline beyond it,
# so the dashboard shows the actual hump shape rather than a misleading monotonic climb.
PRICE_SWEEP = [2.50, 3.86, 5.25, 7.75, 10.50, 14.00]

KAPPA = 0.05
N_SIMULATIONS = 1000
RANDOM_SEED = 42

PITY_SWEEP = [None, 30, 60, 90, 120, 150]

REWARD_TIERS = {
    'Common': {'drop_rate': 0.50, 'value': {'Whale': 30, 'Casual': 15, 'F2P': 8}},
    'Rare': {'drop_rate': 0.15, 'value': {'Whale': 150, 'Casual': 60, 'F2P': 25}},
    'Epic': {'drop_rate': 0.04, 'value': {'Whale': 400, 'Casual': 180, 'F2P': 60}},
    'Legendary': {'drop_rate': 0.01, 'value': {'Whale': 1000, 'Casual': 300, 'F2P': 100}},
}

PLAYER_TYPES = {
    'Whale': {'f_range': (0.03, 0.08), 'alpha': 0.05},
    'Casual': {'f_range': (0.10, 0.20), 'alpha': 0.30},
    'F2P': {'f_range': (0.40, 0.60), 'alpha': 0.65},
}

# ============================================================================
# UTILITY FUNCTIONS (price p is now a true variable, not just a default)
# ============================================================================

def expected_value_multitier(x, player_type, pity_threshold=None):
    if x <= 0:
        return 0
    total_ev = 0
    for tier_name, tier_data in REWARD_TIERS.items():
        r = tier_data['drop_rate']
        v = tier_data['value'][player_type]
        if tier_name == 'Legendary' and pity_threshold is not None and x >= pity_threshold:
            prob = 1.0
        else:
            prob = 1 - (1 - r) ** x
        total_ev += v * prob
    return total_ev


def player_utility(x, player_type, f, p, pity_threshold=None):
    if x <= 0:
        return 0
    ev = expected_value_multitier(x, player_type, pity_threshold)
    cost = p * x
    fatigue_penalty = f * x ** 2
    return ev - cost - fatigue_penalty


def find_optimal_boxes(player_type, f, p, pity_threshold=None):
    upper_bound = (pity_threshold + 50) if pity_threshold is not None else 200

    def objective(x):
        return -player_utility(x, player_type, f, p, pity_threshold)

    result = minimize_scalar(objective, bounds=(0, upper_bound), method='bounded')
    if result.fun >= 0:
        return 0, 0
    return result.x, -result.fun


def tier_revenue_breakdown(player_type, x, pity_threshold=None):
    breakdown = {}
    for tier_name, tier_data in REWARD_TIERS.items():
        r = tier_data['drop_rate']
        v = tier_data['value'][player_type]
        if x <= 0:
            breakdown[tier_name] = 0
            continue
        if tier_name == 'Legendary' and pity_threshold is not None and x >= pity_threshold:
            prob = 1.0
        else:
            prob = 1 - (1 - r) ** x
        breakdown[tier_name] = v * prob
    return breakdown


# ============================================================================
# PRICE x PITY SWEEP
# ============================================================================

def run_sweep():
    player_rows = []
    tier_rows = []

    combo_idx = 0
    for price in PRICE_SWEEP:
        for pity_threshold in PITY_SWEEP:
            pity_label = pity_threshold if pity_threshold is not None else 0
            desc = f"Price=${price} Pity={pity_label}"

            rng = np.random.default_rng(RANDOM_SEED + combo_idx * 1000)
            combo_idx += 1

            for sim in tqdm(range(N_SIMULATIONS), desc=desc):
                for ptype, params in PLAYER_TYPES.items():
                    f = rng.uniform(*params['f_range'])
                    alpha = params['alpha']

                    x_opt, utility = find_optimal_boxes(ptype, f, price, pity_threshold)
                    revenue = alpha * price * x_opt
                    boxes_weighted = alpha * x_opt
                    dev_cost = KAPPA * pity_threshold if pity_threshold else 0

                    player_rows.append({
                        'sim_id': sim,
                        'price_per_box': price,
                        'pity_threshold': pity_label,
                        'player_type': ptype,
                        'population_share': alpha,
                        'fatigue': f,
                        'boxes_opened': x_opt,
                        'boxes_opened_weighted': boxes_weighted,
                        'utility': utility,
                        'revenue': revenue,
                        'dev_cost_allocated': dev_cost * alpha,
                        'net_revenue': revenue - (dev_cost * alpha),
                        'participated': 1 if x_opt > 0 else 0,
                    })

                    tiers = tier_revenue_breakdown(ptype, x_opt, pity_threshold)
                    for tier_name, tier_value in tiers.items():
                        tier_rows.append({
                            'sim_id': sim,
                            'price_per_box': price,
                            'pity_threshold': pity_label,
                            'player_type': ptype,
                            'tier': tier_name,
                            'drop_rate': REWARD_TIERS[tier_name]['drop_rate'],
                            'tier_revenue_contribution': tier_value * alpha,
                        })

    return pd.DataFrame(player_rows), pd.DataFrame(tier_rows)


if __name__ == "__main__":
    print(f"Running sweep: {len(PRICE_SWEEP)} prices x {len(PITY_SWEEP)} pity thresholds "
          f"x {N_SIMULATIONS} sims x {len(PLAYER_TYPES)} player types")
    df_players, df_tiers = run_sweep()

    df_players.to_csv('loot_box_player_panel.csv', index=False)
    df_tiers.to_csv('loot_box_tier_panel.csv', index=False)

    print(f"\nPlayer panel: {len(df_players):,} rows -> loot_box_player_panel.csv")
    print(f"Tier panel:   {len(df_tiers):,} rows -> loot_box_tier_panel.csv")

    print("\nNet revenue heatmap (sum across sims), Price (rows) x Pity Threshold (cols):")
    pivot = df_players.pivot_table(
        index='price_per_box', columns='pity_threshold',
        values='net_revenue', aggfunc='sum'
    ).round(0)
    print(pivot)

    print("\nBest price/pity combo by total net revenue:")
    totals = df_players.groupby(['price_per_box', 'pity_threshold'])['net_revenue'].sum()
    print(totals.idxmax(), '->', round(totals.max(), 2))
