"""
run_nla.py — Feed layer 20 activations through NLA activation verbalizer.

Setup (run once on Lambda before this script):

    # 1. Get nla_inference.py (single file, not a pip package)
    wget https://raw.githubusercontent.com/kitft/nla-inference/main/nla_inference.py

    # 2. Download NLA verbalizer checkpoint locally (NLAClient needs local path + nla_meta.yaml)
    huggingface-cli download kitft/nla-qwen2.5-7b-L20-av --local-dir ./actor_hf

    # 3. Install SGLang
    pip install "sglang[all]>=0.5.6"

    # 4. Launch SGLang server in a separate tmux window
    tmux new -s sglang
    python -m sglang.launch_server \\
        --model-path ./actor_hf \\
        --disable-radix-cache \\
        --mem-fraction-static 0.85 \\
        --trust-remote-code \\
        --port 30000
    # Wait for "Server is ready" before running this script (Ctrl+B D to detach)

Usage:
    tmux new -s nla
    python scripts/run_nla.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# nla_inference.py must be in the project root (wget from kitft/nla-inference)
sys.path.insert(0, str(Path(__file__).parent.parent))
from nla_inference import NLAClient


# ── Config ────────────────────────────────────────────────────────────────────

ACTOR_DIR     = Path('./actor_hf')           # local checkpoint dir with nla_meta.yaml
SGLANG_URL    = 'http://localhost:30000'
ACTIVATIONS   = Path('results/activations_layer20.npz')
OUTPUT_PATH   = Path('results/nla_descriptions.csv')
SAVE_EVERY    = 50                           # checkpoint every N descriptions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load activations
    print(f'Loading {ACTIVATIONS}...')
    data = np.load(ACTIVATIONS, allow_pickle=True)
    scenario_ids = data['scenario_ids']   # (496,)
    activations  = data['activations']   # (496, 3584) float32
    tiers        = data['tiers']
    labels       = data['labels']
    print(f'Loaded {len(activations)} activations, shape {activations.shape}')

    # Crash-resume
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing['scenario_id'].tolist())
        print(f'Resuming: {len(done_ids)} descriptions already generated.')
    else:
        existing = pd.DataFrame()
        done_ids = set()

    pending_mask   = ~np.isin(scenario_ids, list(done_ids))
    pending_ids    = scenario_ids[pending_mask]
    pending_acts   = activations[pending_mask]
    pending_tiers  = tiers[pending_mask]
    pending_labels = labels[pending_mask]
    print(f'Descriptions to generate: {len(pending_ids)}')

    if len(pending_ids) == 0:
        print('All descriptions already generated. Nothing to do.')
        _verify()
        return

    # Init NLA client — loads embedding weights only (~300MB), full model is in SGLang
    print(f'Connecting to SGLang at {SGLANG_URL}...')
    client = NLAClient(str(ACTOR_DIR), sglang_url=SGLANG_URL)
    print('Connected.')

    # Generate with incremental saves
    rows = []
    for start in range(0, len(pending_ids), SAVE_EVERY):
        end = min(start + SAVE_EVERY, len(pending_ids))

        print(f'Generating {start}–{end} of {len(pending_ids)}...')
        descriptions = client.generate_batch(pending_acts[start:end])

        for i, desc in enumerate(descriptions):
            rows.append({
                'scenario_id': int(pending_ids[start + i]),
                'tier':        pending_tiers[start + i],
                'label':       pending_labels[start + i],
                'description': desc,
            })

        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f'  Checkpoint: {len(combined)} total saved')

    print(f'\nDone. {len(combined)} descriptions saved to {OUTPUT_PATH}')
    _verify()


def _verify():
    print('\n' + '=' * 50)
    print('Verification')
    print('=' * 50)
    df = pd.read_csv(OUTPUT_PATH)
    print(f'Total rows:           {len(df)}')
    print(f'Missing descriptions: {df["description"].isna().sum()}')
    print(f'Empty descriptions:   {(df["description"].str.strip() == "").sum()}')
    print('\nTier distribution:')
    print(df['tier'].value_counts().sort_index().to_string())
    print('\nLabel distribution:')
    print(df['label'].value_counts().to_string())
    print('\nSample descriptions:')
    for _, row in df.sample(3, random_state=42).iterrows():
        print(f'\n  id={row["scenario_id"]}  tier={row["tier"]}  label={row["label"]}')
        print(f'  {str(row["description"])[:300]}')


if __name__ == '__main__':
    main()
