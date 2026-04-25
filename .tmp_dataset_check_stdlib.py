from pathlib import Path
import zipfile
import struct
import ast
import csv
from collections import Counter

def npy_shape_from_bytes(b: bytes):
    if len(b) < 10 or b[:6] != b'\x93NUMPY':
        return None
    major = b[6]
    minor = b[7]
    if major == 1:
        hlen = struct.unpack('<H', b[8:10])[0]
        hstart = 10
    else:
        hlen = struct.unpack('<I', b[8:12])[0]
        hstart = 12
    h = b[hstart:hstart+hlen].decode('latin1').strip()
    try:
        d = ast.literal_eval(h)
        return d.get('shape')
    except Exception:
        return None

root = Path('Data')
norm_dir = root / 'processed' / 'new_model' / 'normalized'
raw_dir = root / 'raw'

print('=== 1) One normalized NPZ per signer: keys and shapes ===')
for signer in ['S02','S04','S05','S10']:
    files = sorted(norm_dir.glob(f'*__{signer}__*.npz'))
    if not files:
        print(f'{signer}: NO normalized .npz found')
        continue
    f = files[0]
    print(f'{signer}: {f.name}')
    with zipfile.ZipFile(f, 'r') as zf:
        for n in zf.namelist():
            if not n.endswith('.npy'):
                continue
            key = Path(n).stem
            with zf.open(n) as fp:
                header = fp.read(1024)
            shape = npy_shape_from_bytes(header)
            print(f'  - {key}: shape={shape}')

print('\n=== 2) Raw vs normalized counts per non-S02 signer (filename coverage) ===')
missing_by = {}
for signer in ['S04','S05','S10']:
    raw_bases = {p.stem for p in raw_dir.glob(f'*__{signer}__*.mp4')}
    norm_bases = {p.stem for p in norm_dir.glob(f'*__{signer}__*.npz')}
    matched = raw_bases.intersection(norm_bases)
    missing = sorted(raw_bases - norm_bases)
    extra = sorted(norm_bases - raw_bases)
    missing_by[signer] = missing
    print(f'{signer}: raw={len(raw_bases)}, normalized={len(norm_bases)}, matched_by_filename={len(matched)}, missing_norm_for_raw={len(missing)}, normalized_without_raw={len(extra)}')
    if missing:
        print('  missing examples: ' + '; '.join(missing[:5]))
    if extra:
        print('  extra normalized examples: ' + '; '.join(extra[:5]))

print('\n=== 3) Any raw non-S02 missing corresponding normalized npz? ===')
total_missing = sum(len(v) for v in missing_by.values())
if total_missing:
    print('YES. Missing total=' + str(total_missing) + ' across signers: ' + ', '.join(f'{s}:{len(missing_by[s])}' for s in ['S04','S05','S10']))
else:
    print('NO. Full coverage for S04/S05/S10 raw files.')

print('\n=== 4) filename_rename_map.csv status stats ===')
map_path = root / 'processed' / 'new_model' / 'filename_rename_map.csv'
with map_path.open('r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
cols = reader.fieldnames or []
print('rows_total=' + str(len(rows)))
status_col = None
for c in cols:
    if c and c.strip().lower() == 'status':
        status_col = c
        break
if status_col is None:
    for c in cols:
        if c and 'status' in c.lower():
            status_col = c
            break
if status_col is None:
    print('No status-like column found. Columns: ' + ', '.join(cols))
else:
    print('status_column=' + status_col)
    ctr = Counter((r.get(status_col) or 'NA') for r in rows)
    for k, v in ctr.most_common():
        print(f'{k}: {v}')
