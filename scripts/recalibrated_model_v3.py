"""
RECALIBRATED MODEL v3 — Empirically-Grounded Loot Box Simulation
==================================================================
Two targeted fixes to close the largest theory-vs-reality gaps,
while DELIBERATELY leaving structural limitations intact (documented
honestly rather than force-fit).

FIX 1 — Within-tier price variance (the "Dragon Lore problem"):
  The original model assigned every item in a tier a single fixed
  value. Real Steam Market data shows 6-18x price spread WITHIN each
  tier (a $3 Covert skin and a $1,053 Covert skin coexist). Instead
  of fixed tier values, this version samples each reward draw from a
  lognormal distribution fitted to REAL Steam Market prices per tier.

FIX 2 — Realistic F2P participation:
  The original model had ~100% F2P participation, which is empirically
  false -- most free-to-play players never make a purchase. Industry
  data (Swrve/Everyplay) shows ~2-5% of F2P players spend in a given
  month. This version gates F2P participation behind a realistic
  spend-conversion probability.

DELIBERATELY NOT CHANGED (documented as limitations, not bugs):
  - Multi-session dynamics: model remains one-shot. Real players reset
    fatigue across sessions and chase new collections over weeks. Fixing
    this requires a multi-period rebuild = new research, not a recalibration.
  - Secondary-market feedback loop: skin resale value feeding back into
    willingness-to-pay is not modeled. Structural change, out of scope.
  - Whale revenue concentration: left at model-implied level. Force-fitting
    parameters to hit the literature's ~50% would be curve-fitting to a
    single statistic. The gap is reported as a finding, not hidden.

Author: Jason (recalibration layer for MATH 111A portfolio project)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# ============================================================================
# REAL PRICE DISTRIBUTIONS (fitted from real_case_item_prices.csv)
# Lognormal params (log-mean, log-std) per tier, from actual Steam Market data
# ============================================================================
# Lognormal params per tier, fitted to 129 REAL active-case Steam Market
# prices across ALL four tiers (consistent sample, n~32 per tier). Replaces
# the earlier mixed sample (legacy + active) which systematically overstated
# prices. Clean monotonic structure: Common $0.40 -> Rare $1.15 -> Epic $7.25
# -> Legendary $65 (medians).
REAL_PRICE_LOGNORMAL = {
    'Common':    {'log_mean': -1.08, 'log_std': 0.90},
    'Rare':      {'log_mean': 0.51, 'log_std': 0.95},
    'Epic':      {'log_mean': 2.26, 'log_std': 0.70},
    'Legendary': {'log_mean': 4.39, 'log_std': 0.67},
}

# Winsorization cap: clip sampled values at the 99th percentile of their
# tier distribution so a single extreme draw can't dominate aggregate
# revenue (standard technique for fat-tailed data fit on limited samples).
WINSORIZE_PERCENTILE = 99

# Player-type valuation multipliers applied to REAL market prices.
# These encode willingness-to-pay ABOVE raw resale value: a status good is
# worth more to own than its market price (prestige, identity, rarity).
# Base level (13x) calibrated so aggregate engagement reproduces the paper's
# validated ~8.2 boxes/player (industry norm 5-15), NOT tuned to any revenue
# target. Relative ordering (Whale > Casual > F2P) preserved from the paper.
_BASE_MULT = 13
TYPE_VALUE_MULTIPLIER = {'Whale': _BASE_MULT * 1.4,
                         'Casual': _BASE_MULT * 0.7,
                         'F2P': _BASE_MULT * 0.25}

# F2P spend-conversion: fraction of F2P players who EVER open a box.
# ~3% per industry data (Swrve 2016: ~1.5-2.3%; we use 3% as a generous
# upper bound to avoid overcorrecting).
F2P_CONVERSION_RATE = 0.03

PRICE_PER_BOX = 3.86  # baseline; sweep handled separately if needed
KAPPA = 0.05
N_SIMULATIONS = 1000
RANDOM_SEED = 42

PITY_SWEEP = [None, 30, 60, 90, 120, 150]

REWARD_TIERS = {
    'Common': {'drop_rate': 0.50},
    'Rare': {'drop_rate': 0.15},
    'Epic': {'drop_rate': 0.04},
    'Legendary': {'drop_rate': 0.01},
}

PLAYER_TYPES = {
    'Whale': {'f_range': (0.03, 0.08), 'alpha': 0.05},
    'Casual': {'f_range': (0.10, 0.20), 'alpha': 0.30},
    'F2P': {'f_range': (0.40, 0.60), 'alpha': 0.65},
}


def sample_tier_value(tier, player_type, rng):
    """Sample a reward value from the REAL price distribution for this tier,
    scaled by the player type's subjective valuation multiplier. Winsorized
    at the 99th percentile to prevent single extreme draws from dominating."""
    params = REAL_PRICE_LOGNORMAL[tier]
    market_price = rng.lognormal(params['log_mean'], params['log_std'])
    # cap at 99th percentile of this tier's lognormal
    from scipy.stats import lognorm
    cap = lognorm.ppf(WINSORIZE_PERCENTILE / 100, s=params['log_std'],
                      scale=np.exp(params['log_mean']))
    market_price = min(market_price, cap)
    return market_price * TYPE_VALUE_MULTIPLIER[player_type]


def expected_value_multitier(x, tier_values, pity_threshold=None):
    if x <= 0:
        return 0
    total_ev = 0
    for tier_name, tier_data in REWARD_TIERS.items():
        r = tier_data['drop_rate']
        v = tier_values[tier_name]
        if tier_name == 'Legendary' and pity_threshold is not None and x >= pity_threshold:
            prob = 1.0
        else:
            prob = 1 - (1 - r) ** x
        total_ev += v * prob
    return total_ev


def player_utility(x, tier_values, f, p, pity_threshold=None):
    if x <= 0:
        return 0
    ev = expected_value_multitier(x, tier_values, pity_threshold)
    return ev - p * x - f * x ** 2


def find_optimal_boxes(tier_values, f, p, pity_threshold=None):
    upper_bound = (pity_threshold + 50) if pity_threshold is not None else 200

    def objective(x):
        return -player_utility(x, tier_values, f, p, pity_threshold)

    result = minimize_scalar(objective, bounds=(0, upper_bound), method='bounded')
    if result.fun >= 0:
        return 0, 0
    return result.x, -result.fun


def tier_revenue_breakdown(tier_values, x, pity_threshold=None):
    breakdown = {}
    for tier_name, tier_data in REWARD_TIERS.items():
        r = tier_data['drop_rate']
        v = tier_values[tier_name]
        if x <= 0:
            breakdown[tier_name] = 0
            continue
        if tier_name == 'Legendary' and pity_threshold is not None and x >= pity_threshold:
            prob = 1.0
        else:
            prob = 1 - (1 - r) ** x
        breakdown[tier_name] = v * prob
    return breakdown


def run_sweep():
    player_rows = []
    tier_rows = []

    for pity_threshold in PITY_SWEEP:
        pity_label = pity_threshold if pity_threshold is not None else 0
        rng = np.random.default_rng(RANDOM_SEED + (pity_label or 0))

        for sim in tqdm(range(N_SIMULATIONS), desc=f"Pity={pity_label}"):
            for ptype, params in PLAYER_TYPES.items():
                f = rng.uniform(*params['f_range'])
                alpha = params['alpha']

                # FIX 2: F2P conversion gate — most F2P players never participate
                if ptype == 'F2P' and rng.random() > F2P_CONVERSION_RATE:
                    # non-converting F2P: record a zero-participation row
                    player_rows.append({
                        'sim_id': sim, 'pity_threshold': pity_label, 'player_type': ptype,
                        'population_share': alpha, 'fatigue': f, 'boxes_opened': 0,
                        'boxes_opened_weighted': 0, 'utility': 0, 'revenue': 0,
                        'dev_cost_allocated': 0, 'net_revenue': 0, 'participated': 0,
                    })
                    continue

                # FIX 1: sample reward values from REAL per-tier price distributions
                tier_values = {t: sample_tier_value(t, ptype, rng) for t in REWARD_TIERS}

                x_opt, utility = find_optimal_boxes(tier_values, f, PRICE_PER_BOX, pity_threshold)
                revenue = alpha * PRICE_PER_BOX * x_opt
                dev_cost = KAPPA * pity_threshold if pity_threshold else 0

                player_rows.append({
                    'sim_id': sim, 'pity_threshold': pity_label, 'player_type': ptype,
                    'population_share': alpha, 'fatigue': f, 'boxes_opened': x_opt,
                    'boxes_opened_weighted': alpha * x_opt, 'utility': utility,
                    'revenue': revenue, 'dev_cost_allocated': dev_cost * alpha,
                    'net_revenue': revenue - (dev_cost * alpha),
                    'participated': 1 if x_opt > 0 else 0,
                })

                tiers = tier_revenue_breakdown(tier_values, x_opt, pity_threshold)
                for tier_name, tier_value in tiers.items():
                    tier_rows.append({
                        'sim_id': sim, 'pity_threshold': pity_label, 'player_type': ptype,
                        'tier': tier_name, 'drop_rate': REWARD_TIERS[tier_name]['drop_rate'],
                        'tier_revenue_contribution': tier_value * alpha,
                    })

    return pd.DataFrame(player_rows), pd.DataFrame(tier_rows)


if __name__ == "__main__":
    print("Running recalibrated sweep (real price variance + F2P conversion)...")
    df_players, df_tiers = run_sweep()

    df_players.to_csv('loot_box_player_panel_v3.csv', index=False)
    df_tiers.to_csv('loot_box_tier_panel_v3.csv', index=False)

    print(f"\nPlayer panel: {len(df_players):,} rows")
    print(f"Tier panel:   {len(df_tiers):,} rows")

    # Validation: whale revenue share and F2P participation now
    print("\n--- Recalibration check ---")
    rev_by_type = df_players.groupby('player_type')['revenue'].sum()
    total = rev_by_type.sum()
    print("Revenue share by player type:")
    for t in ['Whale', 'Casual', 'F2P']:
        print(f"  {t:8} {rev_by_type[t]/total*100:.1f}%")

    f2p = df_players[df_players['player_type'] == 'F2P']
    print(f"\nF2P participation rate: {f2p['participated'].mean()*100:.1f}% (was ~100%)")

    # Tier revenue share now
    tier_share = df_tiers.groupby('tier')['tier_revenue_contribution'].sum()
    tshare = (tier_share / tier_share.sum() * 100).round(1)
    print("\nTier revenue share (model v3):")
    for t in ['Common', 'Rare', 'Epic', 'Legendary']:
        print(f"  {t:10} {tshare[t]}%")
