"""
REAL-WORLD VALIDATION DATASET — CS2 Case Economics
====================================================
Pulls real case contents (ByMykel/CSGO-API, public/no-auth) and real
Steam Community Market prices (Valve's own official priceoverview
endpoint, public/no-auth) to compute REAL expected value per rarity
tier, for direct comparison against the theoretical model's predicted
tier revenue shares and optimal price.

No individual user data is touched anywhere in this pipeline --
everything here is aggregate market data and publicly disclosed
drop rates.

Methodology notes (documented honestly, not hidden):
- Within a rarity tier, items are assumed equally likely (the standard
  community assumption used by tools like CSGOFloat/Csgo-Case-Data,
  since Valve has not disclosed item-level weighting within a tier).
- The "contains_rare" knife/glove pool returned by the API is NOT
  case-specific in this dataset (verified: 100% identical across
  cases sampled) -- so it's treated as one shared reference pool for
  the Legendary-tier price estimate, not a per-case value. This is a
  real limitation of the underlying open dataset, stated here rather
  than presented as more precise than it is.
- Prices are Field-Tested, non-StatTrak, lowest listed price -- a
  single representative condition, not a wear-weighted average.

Author: Jason (real-world validation layer for MATH 111A portfolio project)
"""

import json
import time
import urllib.parse
import urllib.request
import pandas as pd

CASE_SAMPLE = [
    'CS:GO Weapon Case 3',
    'Operation Bravo Case',
    'Chroma Case',
    'Spectrum Case',
]
N_KNIFE_SAMPLE = 10  # shared reference pool, not case-specific (see note above)

# Official Valve-disclosed rarity drop rates (2017, via Perfect World / Chinese
# regulatory disclosure). These are constants, not fetched -- they're publicly
# documented facts, same ones referenced in the original paper.
REAL_DROP_RATES = {
    'Mil-Spec': 0.7992,
    'Restricted': 0.1598,
    'Classified': 0.0320,
    'Covert': 0.0064,
    'Knife/Glove': 0.0026,
}

# Mapping from real CS:GO rarity names to the model's tier names
TIER_MAP = {
    'Mil-Spec Grade': 'Common',
    'Restricted': 'Rare',
    'Classified': 'Epic',
    'Covert': 'Legendary',  # Covert + Knife/Glove both roll up to Legendary
}

REQUEST_DELAY = 1.3  # seconds between Steam requests -- deliberately conservative


def get_price(market_hash_name, retries=2):
    """Query Steam's official public priceoverview endpoint. Returns float or None."""
    url = (
        "https://steamcommunity.com/market/priceoverview/?appid=730&currency=1"
        f"&market_hash_name={urllib.parse.quote(market_hash_name)}"
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            if data.get('success') and data.get('lowest_price'):
                price_str = data['lowest_price'].replace('$', '').replace(',', '')
                return float(price_str)
            return None
        except Exception as e:
            if attempt == retries:
                print(f"  [FAILED] {market_hash_name}: {e}")
                return None
            time.sleep(2)
    return None


def main():
    crates = json.load(open('/tmp/crates.json'))

    rows = []  # one row per priced item: case, tier, item_name, price

    # --- Knife/glove shared reference pool (priced once) ---
    sample_case = next(c for c in crates if c['name'] == CASE_SAMPLE[0])
    knife_sample = sample_case['contains_rare'][:N_KNIFE_SAMPLE]
    print(f"Pricing {len(knife_sample)} reference knives/gloves (shared pool)...")
    knife_prices = []
    for item in knife_sample:
        name = f"{item['name']} (Field-Tested)"
        price = get_price(name)
        time.sleep(REQUEST_DELAY)
        if price is None:
            # vanilla knives are often "no wear" items, retry without condition suffix
            price = get_price(item['name'])
            time.sleep(REQUEST_DELAY)
        print(f"  {item['name']}: {'$'+str(price) if price else 'no price found'}")
        if price is not None:
            knife_prices.append(price)

    avg_knife_price = sum(knife_prices) / len(knife_prices) if knife_prices else None
    print(f"-> Avg knife/glove reference price: ${avg_knife_price:.2f}\n" if avg_knife_price else "-> No knife prices found\n")

    # --- Per-case contents ---
    for case_name in CASE_SAMPLE:
        case = next(c for c in crates if c['name'] == case_name)
        print(f"Pricing {case_name} ({len(case['contains'])} items)...")

        for item in case['contains']:
            real_rarity = item['rarity']['name']
            model_tier = TIER_MAP.get(real_rarity)
            name = f"{item['name']} (Field-Tested)"
            price = get_price(name)
            time.sleep(REQUEST_DELAY)
            print(f"  [{real_rarity}] {item['name']}: {'$'+str(price) if price else 'no price'}")
            rows.append({
                'case': case_name,
                'item_name': item['name'],
                'real_rarity': real_rarity,
                'model_tier': model_tier,
                'price_usd': price,
            })

        # case's own market price (cost of the case itself, key bought separately)
        case_price = get_price(case_name)
        time.sleep(REQUEST_DELAY)
        print(f"  [case price] {case_name}: {'$'+str(case_price) if case_price else 'no price'}")
        rows.append({
            'case': case_name,
            'item_name': case_name,
            'real_rarity': 'Case (container)',
            'model_tier': None,
            'price_usd': case_price,
        })

    df_items = pd.DataFrame(rows)
    df_items.to_csv('real_case_item_prices.csv', index=False)
    print(f"\nSaved {len(df_items)} item price rows -> real_case_item_prices.csv")

    # --- Compute real tier EV per case ---
    tier_rows = []
    for case_name in CASE_SAMPLE:
        case_items = df_items[(df_items['case'] == case_name) & (df_items['model_tier'].notna())]
        for model_tier, real_rarity_name in [('Common', 'Mil-Spec Grade'), ('Rare', 'Restricted'),
                                              ('Epic', 'Classified'), ('Legendary', 'Covert')]:
            tier_items = case_items[case_items['real_rarity'] == real_rarity_name]
            valid_prices = tier_items['price_usd'].dropna()
            avg_price = valid_prices.mean() if len(valid_prices) > 0 else None

            if model_tier == 'Legendary':
                # blend the case's own Covert item price with the shared knife/glove
                # pool, weighted by their real relative drop rates within "Legendary"
                covert_rate, knife_rate = REAL_DROP_RATES['Covert'], REAL_DROP_RATES['Knife/Glove']
                total = covert_rate + knife_rate
                if avg_price is not None and avg_knife_price is not None:
                    avg_price = (avg_price * covert_rate + avg_knife_price * knife_rate) / total
                elif avg_knife_price is not None:
                    avg_price = avg_knife_price

            real_tier_drop_rate = (REAL_DROP_RATES['Covert'] + REAL_DROP_RATES['Knife/Glove']
                                    if model_tier == 'Legendary'
                                    else REAL_DROP_RATES[real_rarity_name.replace(' Grade', '')])

            tier_rows.append({
                'case': case_name,
                'model_tier': model_tier,
                'real_rarity_name': real_rarity_name,
                'real_drop_rate': real_tier_drop_rate,
                'avg_item_price_usd': avg_price,
                'tier_ev_contribution': (avg_price * real_tier_drop_rate) if avg_price is not None else None,
            })

    df_tiers = pd.DataFrame(tier_rows)
    df_tiers.to_csv('real_case_tier_ev.csv', index=False)
    print(f"Saved tier EV breakdown -> real_case_tier_ev.csv\n")
    print(df_tiers.to_string(index=False))


if __name__ == "__main__":
    main()
