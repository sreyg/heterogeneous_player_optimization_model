"""
IMPROVED LOOT BOX MODEL
=======================
Enhancements:
1. Heterogeneous fatigue rates (Monte Carlo simulation)
2. Multi-tier reward system (Common/Rare/Epic/Legendary)
3. Confidence intervals and distributional analysis
4. Revenue decomposition by tier

Author: Jason (MATH 111A Project)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pandas as pd
from tqdm import tqdm
import seaborn as sns

# Set style for professional plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PARAMETERS
# ============================================================================

# Pricing
PRICE_PER_BOX = 3.86

# Development costs
C_COST = 10
KAPPA = 0.05

# Monte Carlo parameters
N_SIMULATIONS = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# MULTI-TIER REWARD STRUCTURE
# ============================================================================

REWARD_TIERS = {
    'Common': {
        'drop_rate': 0.50,  # 50% chance per box
        'value': {
            'Whale': 30,
            'Casual': 15,
            'F2P': 8
        },
        'color': '#95E1D3'
    },
    'Rare': {
        'drop_rate': 0.15,  # 15% chance per box
        'value': {
            'Whale': 150,
            'Casual': 60,
            'F2P': 25
        },
        'color': '#4ECDC4'
    },
    'Epic': {
        'drop_rate': 0.04,  # 4% chance per box
        'value': {
            'Whale': 400,
            'Casual': 180,
            'F2P': 60
        },
        'color': '#FF6B6B'
    },
    'Legendary': {
        'drop_rate': 0.01,  # 1% chance per box
        'value': {
            'Whale': 1000,
            'Casual': 300,
            'F2P': 100
        },
        'color': '#FFD93D'
    }
}

# ============================================================================
# PLAYER TYPE PARAMETERS WITH HETEROGENEOUS FATIGUE
# ============================================================================

PLAYER_TYPES = {
    'Whale': {
        'v_base': 1000,  # Base valuation (will use multi-tier)
        'f_range': (0.03, 0.08),  # Fatigue varies in this range
        'alpha': 0.05,
        'color': '#FF6B6B'
    },
    'Casual': {
        'v_base': 300,
        'f_range': (0.10, 0.20),
        'alpha': 0.30,
        'color': '#4ECDC4'
    },
    'F2P': {
        'v_base': 100,
        'f_range': (0.40, 0.60),
        'alpha': 0.65,
        'color': '#95E1D3'
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def expected_value_multitier(x, player_type):
    """
    Calculate expected value from opening x boxes with multi-tier rewards.
    
    Args:
        x: number of boxes to open
        player_type: 'Whale', 'Casual', or 'F2P'
    
    Returns:
        Expected value considering all tiers
    """
    if x <= 0:
        return 0
    
    total_ev = 0
    for tier_name, tier_data in REWARD_TIERS.items():
        r = tier_data['drop_rate']
        v = tier_data['value'][player_type]
        # Probability of getting at least one of this tier in x boxes
        prob = 1 - (1 - r)**x
        total_ev += v * prob
    
    return total_ev


def player_utility_multitier(x, player_type, f, p=PRICE_PER_BOX):
    """
    Player utility with multi-tier rewards and individual fatigue.
    
    Args:
        x: number of boxes
        player_type: 'Whale', 'Casual', or 'F2P'
        f: individual fatigue coefficient
        p: price per box
    
    Returns:
        Utility value
    """
    if x <= 0:
        return 0
    
    expected_value = expected_value_multitier(x, player_type)
    cost = p * x
    fatigue_penalty = f * x**2
    
    return expected_value - cost - fatigue_penalty


def find_optimal_boxes(player_type, f, p=PRICE_PER_BOX):
    """
    Find optimal number of boxes for a player to open.
    
    Args:
        player_type: 'Whale', 'Casual', or 'F2P'
        f: individual fatigue coefficient
        p: price per box
    
    Returns:
        (optimal_x, max_utility)
    """
    def objective(x):
        return -player_utility_multitier(x, player_type, f, p)
    
    result = minimize_scalar(objective, bounds=(0, 200), method='bounded')
    
    # If utility is negative, player doesn't participate
    if result.fun >= 0:
        return 0, 0
    
    return result.x, -result.fun


# ============================================================================
# PITY SYSTEM UTILITIES
# ============================================================================

def expected_value_multitier_pity(x, player_type, pity_threshold):
    """
    Expected value with pity system - guaranteed legendary after N boxes.
    """
    if x <= 0:
        return 0
    
    total_ev = 0
    
    # Add all tiers except legendary (normal probability)
    for tier_name, tier_data in REWARD_TIERS.items():
        if tier_name == 'Legendary':
            # Legendary has pity mechanic
            r = tier_data['drop_rate']
            v = tier_data['value'][player_type]
            if x >= pity_threshold:
                prob = 1.0  # Guaranteed
            else:
                prob = 1 - (1 - r)**x
            total_ev += v * prob
        else:
            # Other tiers work normally
            r = tier_data['drop_rate']
            v = tier_data['value'][player_type]
            prob = 1 - (1 - r)**x
            total_ev += v * prob
    
    return total_ev


def player_utility_pity(x, player_type, f, pity_threshold, p=PRICE_PER_BOX):
    """Player utility with pity system."""
    if x <= 0:
        return 0
    
    expected_value = expected_value_multitier_pity(x, player_type, pity_threshold)
    cost = p * x
    fatigue_penalty = f * x**2
    
    return expected_value - cost - fatigue_penalty


def find_optimal_boxes_pity(player_type, f, pity_threshold, p=PRICE_PER_BOX):
    """Find optimal boxes with pity system."""
    def objective(x):
        return -player_utility_pity(x, player_type, f, pity_threshold, p)
    
    result = minimize_scalar(objective, bounds=(0, pity_threshold + 50), method='bounded')
    
    if result.fun >= 0:
        return 0, 0
    
    return result.x, -result.fun


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo_simulation(n_sims=N_SIMULATIONS, pity_threshold=None):
    """
    Run Monte Carlo simulation with heterogeneous fatigue rates.
    
    Args:
        n_sims: number of simulations
        pity_threshold: if not None, use pity system with this threshold
    
    Returns:
        DataFrame with simulation results
    """
    results = []
    
    print(f"\nRunning Monte Carlo Simulation ({n_sims} iterations)...")
    print(f"Pity system: {'N=' + str(pity_threshold) if pity_threshold else 'None'}")
    print("-" * 70)
    
    for sim in tqdm(range(n_sims), desc="Simulating"):
        # Randomize fatigue for each player type
        fatigue = {
            ptype: np.random.uniform(*params['f_range'])
            for ptype, params in PLAYER_TYPES.items()
        }
        
        # Calculate optimal behavior for each player type
        total_revenue = 0
        total_boxes = 0
        player_data = {}
        
        for ptype, params in PLAYER_TYPES.items():
            f = fatigue[ptype]
            alpha = params['alpha']
            
            if pity_threshold is None:
                x_opt, utility = find_optimal_boxes(ptype, f)
            else:
                x_opt, utility = find_optimal_boxes_pity(ptype, f, pity_threshold)
            
            revenue_contrib = alpha * PRICE_PER_BOX * x_opt
            total_revenue += revenue_contrib
            total_boxes += alpha * x_opt
            
            player_data[ptype] = {
                'boxes': x_opt,
                'utility': utility,
                'revenue': revenue_contrib,
                'fatigue': f
            }
        
        # Calculate developer costs
        if pity_threshold is None:
            # For no pity, we need to estimate effective drop rate
            # Use average tier drop rate weighted by value
            dev_cost = 0  # Simplified - no r^2 term without single drop rate
        else:
            dev_cost = KAPPA * pity_threshold
        
        net_revenue = total_revenue - dev_cost
        
        # Store results
        result = {
            'sim_id': sim,
            'total_revenue': total_revenue,
            'dev_cost': dev_cost,
            'net_revenue': net_revenue,
            'total_boxes': total_boxes,
            'pity': pity_threshold if pity_threshold else 0
        }
        
        # Add player-specific data
        for ptype in PLAYER_TYPES.keys():
            result[f'{ptype}_boxes'] = player_data[ptype]['boxes']
            result[f'{ptype}_utility'] = player_data[ptype]['utility']
            result[f'{ptype}_revenue'] = player_data[ptype]['revenue']
            result[f'{ptype}_fatigue'] = player_data[ptype]['fatigue']
        
        results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# REVENUE BY TIER ANALYSIS
# ============================================================================

def analyze_revenue_by_tier(player_type, x, f):
    """
    Break down expected revenue contribution by each tier.
    
    Returns:
        Dictionary with revenue by tier
    """
    tier_revenues = {}
    
    for tier_name, tier_data in REWARD_TIERS.items():
        r = tier_data['drop_rate']
        v = tier_data['value'][player_type]
        prob = 1 - (1 - r)**x if x > 0 else 0
        tier_revenues[tier_name] = v * prob
    
    return tier_revenues


def calculate_tier_revenue_distribution(df_results):
    """
    Calculate average revenue contribution by tier across all simulations.
    """
    tier_revenues = {tier: [] for tier in REWARD_TIERS.keys()}
    
    for _, row in df_results.iterrows():
        for ptype, params in PLAYER_TYPES.items():
            x = row[f'{ptype}_boxes']
            f = row[f'{ptype}_fatigue']
            alpha = params['alpha']
            
            tier_breakdown = analyze_revenue_by_tier(ptype, x, f)
            
            for tier_name, tier_value in tier_breakdown.items():
                # Weight by population and box opening probability
                tier_revenues[tier_name].append(tier_value * alpha)
    
    # Calculate means
    tier_means = {tier: np.mean(revenues) for tier, revenues in tier_revenues.items()}
    
    return tier_means


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_revenue_distribution(df_results, save_path='revenue_distribution.png'):
    """Plot distribution of net revenues from Monte Carlo simulation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_rev = df_results['net_revenue'].mean()
    std_rev = df_results['net_revenue'].std()
    ci_95 = 1.96 * std_rev
    
    # Histogram
    ax.hist(df_results['net_revenue'], bins=50, alpha=0.7, color='#4ECDC4', 
            edgecolor='black', linewidth=1.2)
    
    # Mean line
    ax.axvline(mean_rev, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: ${mean_rev:.2f}')
    
    # Confidence interval
    ax.axvline(mean_rev - ci_95, color='orange', linestyle=':', linewidth=2,
               label=f'95% CI: ±${ci_95:.2f}')
    ax.axvline(mean_rev + ci_95, color='orange', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Net Revenue per Player ($)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Monte Carlo Revenue Distribution\n(Heterogeneous Fatigue Rates)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig


def plot_player_engagement(df_results, save_path='player_engagement.png'):
    """Plot distribution of boxes opened by player type."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    player_types = ['Whale', 'Casual', 'F2P']
    
    for idx, ptype in enumerate(player_types):
        ax = axes[idx]
        data = df_results[f'{ptype}_boxes']
        color = PLAYER_TYPES[ptype]['color']
        
        mean_boxes = data.mean()
        std_boxes = data.std()
        
        ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(mean_boxes, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_boxes:.1f}')
        
        ax.set_xlabel('Boxes Opened', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{ptype}\n(σ = {std_boxes:.2f})', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig


def plot_tier_revenue_breakdown(tier_revenues, save_path='tier_revenue_breakdown.png'):
    """Plot revenue contribution by tier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    tiers = list(tier_revenues.keys())
    revenues = list(tier_revenues.values())
    colors = [REWARD_TIERS[tier]['color'] for tier in tiers]
    
    # Bar chart
    bars = ax1.bar(tiers, revenues, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Expected Revenue per Player ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Revenue Contribution by Tier', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(revenues, labels=tiers, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Revenue Distribution by Tier', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig


def plot_fatigue_sensitivity(df_results, save_path='fatigue_sensitivity.png'):
    """Plot how revenue varies with fatigue rates."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    player_types = ['Whale', 'Casual', 'F2P']
    
    for idx, ptype in enumerate(player_types):
        ax = axes[idx]
        
        fatigue = df_results[f'{ptype}_fatigue']
        boxes = df_results[f'{ptype}_boxes']
        
        ax.scatter(fatigue, boxes, alpha=0.3, s=20, color=PLAYER_TYPES[ptype]['color'])
        
        # Trend line
        z = np.polyfit(fatigue, boxes, 2)
        p = np.poly1d(z)
        fatigue_range = np.linspace(fatigue.min(), fatigue.max(), 100)
        ax.plot(fatigue_range, p(fatigue_range), 'r-', linewidth=2.5, 
                label='Quadratic fit')
        
        ax.set_xlabel('Fatigue Coefficient', fontsize=12, fontweight='bold')
        ax.set_ylabel('Boxes Opened', fontsize=12)
        ax.set_title(f'{ptype} Sensitivity', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig


def plot_comparison_pity_vs_no_pity(df_no_pity, df_pity, save_path='pity_comparison.png'):
    """Compare results with and without pity system."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Revenue comparison
    data_to_plot = [df_no_pity['net_revenue'], df_pity['net_revenue']]
    labels = ['No Pity', f'Pity (N={int(df_pity["pity"].iloc[0])})']
    
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     medianprops={'color': 'red', 'linewidth': 2})
    
    colors = ['#4ECDC4', '#FF6B6B']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Net Revenue per Player ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Revenue Distribution Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add means as text
    for i, data in enumerate(data_to_plot):
        mean_val = data.mean()
        ax1.text(i+1, mean_val, f'μ=${mean_val:.2f}', ha='center', 
                fontweight='bold', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Engagement comparison
    engagement_no_pity = df_no_pity['total_boxes']
    engagement_pity = df_pity['total_boxes']
    
    ax2.hist(engagement_no_pity, bins=30, alpha=0.6, color='#4ECDC4', 
            label='No Pity', edgecolor='black')
    ax2.hist(engagement_pity, bins=30, alpha=0.6, color='#FF6B6B',
            label=f'Pity (N={int(df_pity["pity"].iloc[0])})', edgecolor='black')
    
    ax2.set_xlabel('Average Boxes Opened per Player', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Engagement Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary_statistics(df_results, label="Base Model"):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 70)
    print(f"SUMMARY STATISTICS: {label}")
    print("=" * 70)
    
    # Overall metrics
    mean_rev = df_results['net_revenue'].mean()
    std_rev = df_results['net_revenue'].std()
    ci_95 = 1.96 * std_rev
    median_rev = df_results['net_revenue'].median()
    
    print(f"\n📊 REVENUE METRICS:")
    print(f"   Mean Net Revenue:    ${mean_rev:.2f}")
    print(f"   Median Net Revenue:  ${median_rev:.2f}")
    print(f"   Std Deviation:       ${std_rev:.2f}")
    print(f"   95% Confidence Int:  [${mean_rev - ci_95:.2f}, ${mean_rev + ci_95:.2f}]")
    
    # Engagement
    mean_boxes = df_results['total_boxes'].mean()
    std_boxes = df_results['total_boxes'].std()
    
    print(f"\n📦 ENGAGEMENT METRICS:")
    print(f"   Mean Boxes/Player:   {mean_boxes:.2f} ± {std_boxes:.2f}")
    
    # Player type breakdown
    print(f"\n👥 PLAYER TYPE BREAKDOWN:")
    print(f"   {'Type':<10} {'Boxes':<15} {'Utility':<15} {'Revenue':<15}")
    print(f"   {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    
    for ptype in ['Whale', 'Casual', 'F2P']:
        mean_boxes_type = df_results[f'{ptype}_boxes'].mean()
        std_boxes_type = df_results[f'{ptype}_boxes'].std()
        mean_utility = df_results[f'{ptype}_utility'].mean()
        mean_revenue = df_results[f'{ptype}_revenue'].mean()
        
        print(f"   {ptype:<10} {mean_boxes_type:>6.2f} ± {std_boxes_type:<5.2f} "
              f"${mean_utility:>6.2f}       ${mean_revenue:>6.2f}")
    
    # Fatigue ranges observed
    print(f"\n🔄 FATIGUE COEFFICIENT RANGES (observed):")
    for ptype in ['Whale', 'Casual', 'F2P']:
        f_min = df_results[f'{ptype}_fatigue'].min()
        f_max = df_results[f'{ptype}_fatigue'].max()
        f_mean = df_results[f'{ptype}_fatigue'].mean()
        print(f"   {ptype:<10} [{f_min:.3f}, {f_max:.3f}]  (μ = {f_mean:.3f})")
    
    print("\n" + "=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("IMPROVED LOOT BOX REVENUE OPTIMIZATION MODEL")
    print("Multi-Tier Rewards + Heterogeneous Fatigue (Monte Carlo)")
    print("="*70)
    
    # Run Monte Carlo without pity
    print("\n\n🎲 SCENARIO 1: NO PITY SYSTEM")
    df_no_pity = run_monte_carlo_simulation(n_sims=N_SIMULATIONS, pity_threshold=None)
    print_summary_statistics(df_no_pity, "No Pity System")
    
    # Run Monte Carlo with pity (N=90)
    print("\n\n🎲 SCENARIO 2: WITH PITY SYSTEM (N=90)")
    df_pity_90 = run_monte_carlo_simulation(n_sims=N_SIMULATIONS, pity_threshold=90)
    print_summary_statistics(df_pity_90, "Pity System (N=90)")
    
    # Analyze tier revenue breakdown
    print("\n\n💰 TIER REVENUE ANALYSIS (No Pity):")
    print("-" * 70)
    tier_revenues = calculate_tier_revenue_distribution(df_no_pity)
    total_tier_rev = sum(tier_revenues.values())
    
    for tier, revenue in tier_revenues.items():
        drop_rate = REWARD_TIERS[tier]['drop_rate']
        pct = (revenue / total_tier_rev * 100) if total_tier_rev > 0 else 0
        print(f"   {tier:<12} Drop Rate: {drop_rate*100:>5.1f}%  →  "
              f"Revenue: ${revenue:>6.2f}  ({pct:>5.1f}% of total)")
    
    print(f"\n   {'Total':<12} {'':>15}    ${total_tier_rev:>6.2f}")
    
    # Generate all visualizations
    print("\n\n📊 GENERATING VISUALIZATIONS...")
    print("-" * 70)
    
    plot_revenue_distribution(df_no_pity, 'revenue_distribution_no_pity.png')
    plot_revenue_distribution(df_pity_90, 'revenue_distribution_pity_90.png')
    plot_player_engagement(df_no_pity, 'player_engagement.png')
    plot_tier_revenue_breakdown(tier_revenues, 'tier_revenue_breakdown.png')
    plot_fatigue_sensitivity(df_no_pity, 'fatigue_sensitivity.png')
    plot_comparison_pity_vs_no_pity(df_no_pity, df_pity_90, 'pity_comparison.png')
    
    # Save results to CSV
    df_no_pity.to_csv('monte_carlo_no_pity.csv', index=False)
    df_pity_90.to_csv('monte_carlo_pity_90.csv', index=False)
    print("\n✓ Saved: monte_carlo_no_pity.csv")
    print("✓ Saved: monte_carlo_pity_90.csv")
    
    # Final comparison
    print("\n\n" + "="*70)
    print("FINAL COMPARISON: PITY VS NO PITY")
    print("="*70)
    
    improvement = df_pity_90['net_revenue'].mean() - df_no_pity['net_revenue'].mean()
    pct_change = (improvement / df_no_pity['net_revenue'].mean()) * 100
    
    print(f"\nNo Pity:     ${df_no_pity['net_revenue'].mean():.2f} "
          f"± ${1.96 * df_no_pity['net_revenue'].std():.2f}")
    print(f"Pity (N=90): ${df_pity_90['net_revenue'].mean():.2f} "
          f"± ${1.96 * df_pity_90['net_revenue'].std():.2f}")
    print(f"\nChange:      ${improvement:.2f} ({pct_change:+.1f}%)")
    
    if improvement > 0:
        print(f"\n✓ Pity system INCREASES revenue by ${improvement:.2f}")
    else:
        print(f"\n✗ Pity system DECREASES revenue by ${-improvement:.2f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • revenue_distribution_no_pity.png")
    print("  • revenue_distribution_pity_90.png")
    print("  • player_engagement.png")
    print("  • tier_revenue_breakdown.png")
    print("  • fatigue_sensitivity.png")
    print("  • pity_comparison.png")
    print("  • monte_carlo_no_pity.csv")
    print("  • monte_carlo_pity_90.csv")
    print("\n🎉 Ready for your presentation slides!\n")
