from pathlib import Path
import numpy as np
import pandas as pd

root = Path('Data')
norm_dir = root / 'processed' / 'new_model' / 'normalized'
raw_dir = root / 'raw'

print('=== 1) One normalized NPZ per signer: keys and shapes ===')
for signer in ['S02', 'S04', 'S05', 'S10']:
    files = sorted(norm_dir.glob(f'*__{signer}__*.npz'))
    if not files:
        print(f'{signer}: NO normalized .npz found')
        continue
    f = files[0]
    z = np.load(f, allow_pickle=True)
    print(f'{signer}: {f.name}')
    for k in z.files:
        v = z[k]
        print(f'  - {k}: shape={getattr(v, "shape", None)}')
    z.close()

print('\n=== 2) Raw vs normalized counts per non-S02 signer (filename coverage) ===')
missing_by_signer = {}
for signer in ['S04', 'S05', 'S10']:
    raw_bases = {p.stem for p in raw_dir.glob(f'*__{signer}__*.mp4')}
    norm_bases = {p.stem for p in norm_dir.glob(f'*__{signer}__*.npz')}
    matched = raw_bases.intersection(norm_bases)
    missing = sorted(raw_bases - norm_bases)
    extra = sorted(norm_bases - raw_bases)
    missing_by_signer[signer] = missing
    print(f'{signer}: raw={len(raw_bases)}, normalized={len(norm_bases)}, matched_by_filename={len(matched)}, missing_norm_for_raw={len(missing)}, normalized_without_raw={len(extra)}')
    if missing:
        print('  missing examples: ' + '; '.join(missing[:5]))
    if extra:
        print('  extra normalized examples: ' + '; '.join(extra[:5]))

print('\n=== 3) Any raw non-S02 missing corresponding normalized npz? ===')
total_missing = sum(len(v) for v in missing_by_signer.values())
if total_missing:
    parts = [f'{s}:{len(missing_by_signer[s])}' for s in ['S04','S05','S10']]
    print('YES. Missing total=' + str(total_missing) + ' across signers: ' + ', '.join(parts))
else:
    print('NO. Full coverage for S04/S05/S10 raw files.')

print('\n=== 4) filename_rename_map.csv status stats ===')
map_path = root / 'processed' / 'new_model' / 'filename_rename_map.csv'
df = pd.read_csv(map_path)
print('rows_total=' + str(len(df)))
status_col = None
for c in df.columns:
    if str(c).strip().lower() == 'status':
        status_col = c
        break
if status_col is None:
    for c in df.columns:
        if 'status' in str(c).lower():
            status_col = c
            break
if status_col is None:
    print('No status-like column found. Columns: ' + ', '.join(map(str, df.columns)))
else:
    print('status_column=' + str(status_col))
    vc = df[status_col].fillna('NA').astype(str).value_counts(dropna=False)
    for k, v in vc.items():
        print(f'{k}: {v}')
