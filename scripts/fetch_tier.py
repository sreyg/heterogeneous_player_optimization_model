import json, time, urllib.parse, urllib.request, random, sys, os
import pandas as pd

random.seed(42)
data = json.load(open('/tmp/crates.json'))

ACTIVE_CASES = ['Kilowatt Case','Recoil Case','Dreams & Nightmares Case','Fracture Case',
    'Snakebite Case','Clutch Case','Danger Zone Case','Prisma Case','Prisma 2 Case',
    'Fever Case','Gallery Case','Revolution Case','Chroma 3 Case','Horizon Case',
    'Shattered Web Case','Glove Case','Operation Riptide Case','Operation Broken Fang Case','CS20 Case']

TIER_MAP = {'Mil-Spec Grade':'Common','Restricted':'Rare','Classified':'Epic','Covert':'Legendary'}
CAP_PER_TIER = 35
OUT = 'real_prices_active_all_tiers.csv'

target_tier = sys.argv[1]  # 'Common','Rare','Epic','Legendary'

# Build pool for the target tier
pool = []
for name in ACTIVE_CASES:
    c = next((c for c in data if c['name']==name), None)
    if not c or not c.get('contains'): continue
    for item in c['contains']:
        if TIER_MAP.get(item['rarity']['name']) == target_tier:
            pool.append(item['name'])
pool = list(dict.fromkeys(pool))
if len(pool) > CAP_PER_TIER:
    pool = random.sample(pool, CAP_PER_TIER)

# Load existing progress
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = set(prev[prev.tier==target_tier]['item'].tolist())

def get_price(name, retries=2):
    url = 'https://steamcommunity.com/market/priceoverview/?appid=730&currency=1&market_hash_name=' + urllib.parse.quote(name)
    for a in range(retries+1):
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                d = json.loads(r.read())
            if d.get('success') and d.get('lowest_price'):
                return float(d['lowest_price'].replace('$','').replace(',',''))
            return None
        except Exception:
            time.sleep(5*(a+1))
    return None

print(f'{target_tier}: {len(pool)} items, {len(done)} already done')
for item in pool:
    if item in done:
        continue
    p = get_price(item + ' (Field-Tested)')
    time.sleep(2.0)
    if p:
        row = pd.DataFrame([{'tier':target_tier,'item':item,'price':p}])
        row.to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        print(f'  {item}: ${p}')
    else:
        print(f'  {item}: no price')
print(f'{target_tier} done.')
