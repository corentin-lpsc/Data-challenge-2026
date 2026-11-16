import re
import numpy as np
import pandas as pd

STRUCT = re.compile(r'^(t|del|inv|add|der|dup|i|ins|r|idic|dic|hsr|tas)\(')
MONO = re.compile(r'^-(\d+|x|y)$')
TRI = re.compile(r'^\+(\d+|x|y)$')

def _clone_tokens(clone):
    """Tokens d'anomalies d'un clone (sans nb de chromosomes, sexe, [n])."""
    clone = re.sub(r'\[\d+\]', '', clone).strip()
    toks = [t.strip() for t in re.split(r',(?![^()]*\))', clone) if t.strip()]
    # toks[0] = nb de chromosomes (46, 45~47...), toks[1] = XX/XY si présent
    rest = toks[1:]
    if rest and re.fullmatch(r'x+y*|x*y+', rest[0]):
        rest = rest[1:]
    return rest

def parse_cyto(s):
    if pd.isna(s) or not str(s).strip():
        return None
    s = str(s).lower().replace(' ', '')
    clones = [c for c in s.split('/') if c]
    counts = [int(m) for m in re.findall(r'\[(\d+)\]', s)]
    stem, all_abn, n_per_clone, abn_cells = [], set(), [], 0
    for i, c in enumerate(clones):
        n = re.search(r'\[(\d+)\]', c)
        toks = _clone_tokens(c)
        if toks and toks[0] in ('idem', 'sl', 'sdl', 'sdl1'):
            toks = stem + toks[1:]
        elif i == 0:
            stem = toks
        toks = [t for t in toks if t not in ('c',)]
        n_per_clone.append(len(toks))
        all_abn.update(toks)
        if toks and n:
            abn_cells += int(n.group(1))
    return dict(tokens=all_abn, n_max=max(n_per_clone) if n_per_clone else 0,
                n_clones=len(clones), abn_frac=abn_cells / sum(counts) if counts else np.nan,
                raw=s)

# Ordre des colonnes fixe (identique entre train et test)
COLUMNS = ['cyto_missing', 'cyto_normal', 'cyto_n_abn', 'cyto_n_clones', 'cyto_abn_frac',
           'cyto_complex', 'cyto_t8_21', 'cyto_inv16', 'cyto_t15_17', 'cyto_minus5_del5q',
           'cyto_minus7', 'cyto_del7q', 'cyto_17p', 'cyto_plus8', 'cyto_inv3', 'cyto_t6_9',
           'cyto_t9_22', 'cyto_t9_11', 'cyto_11q23_other', 'cyto_minus_y', 'cyto_monosomal',
           'cyto_risk']

def cyto_features(series):
    """DataFrame de features cytogénétiques, colonnes toujours dans le même ordre."""
    rows = []
    for s in series:
        p = parse_cyto(s)
        if p is None:
            rows.append({'cyto_missing': 1}); continue
        tk, raw = p['tokens'], p['raw']
        mono = {MONO.match(t).group(1) for t in tk if MONO.match(t)}
        auto_mono = {m for m in mono if m not in ('x', 'y')}
        tri = {TRI.match(t).group(1) for t in tk if TRI.match(t)}
        struct = [t for t in tk if STRUCT.match(t)]
        f = {
            'cyto_missing': 0,
            'cyto_normal': int(len(tk) == 0),
            'cyto_n_abn': p['n_max'],
            'cyto_n_clones': p['n_clones'],
            'cyto_abn_frac': p['abn_frac'],
            'cyto_complex': int(p['n_max'] >= 3),
            'cyto_t8_21': int('t(8;21)' in raw),
            'cyto_inv16': int(bool(re.search(r'inv\(16\)|t\(16;16\)', raw))),
            'cyto_t15_17': int('t(15;17)' in raw),
            'cyto_minus5_del5q': int('5' in mono or bool(re.search(r'del\(5\)\(q', raw))),
            'cyto_minus7': int('7' in mono),
            'cyto_del7q': int(bool(re.search(r'del\(7\)\(q', raw))),
            'cyto_17p': int('17' in mono or bool(re.search(r'del\(17\)\(p|i\(17\)\(q10\)|17p', raw))),
            'cyto_plus8': int('8' in tri),
            'cyto_inv3': int(bool(re.search(r'inv\(3\)|t\(3;3\)', raw))),
            'cyto_t6_9': int('t(6;9)' in raw),
            'cyto_t9_22': int('t(9;22)' in raw),
            'cyto_t9_11': int('t(9;11)' in raw),
            'cyto_11q23_other': int(bool(re.search(r't\([\dxy;]*11[\dxy;]*\)\([^)]*q23', raw))
                                    and 't(9;11)' not in raw),
            'cyto_minus_y': int('y' in mono),
        }
        cbf = f['cyto_t8_21'] or f['cyto_inv16']
        f['cyto_monosomal'] = int(not cbf and (len(auto_mono) >= 2 or
                                  (len(auto_mono) == 1 and len(struct) >= 1)))
        # Catégorie de risque cytogénétique simplifiée (inspirée ELN 2022)
        adverse = (f['cyto_complex'] or f['cyto_monosomal'] or f['cyto_minus5_del5q']
                   or f['cyto_minus7'] or f['cyto_17p'] or f['cyto_inv3']
                   or f['cyto_t6_9'] or f['cyto_t9_22'] or f['cyto_11q23_other'])
        f['cyto_risk'] = 0 if (cbf or f['cyto_t15_17']) else (2 if adverse else 1)
        rows.append(f)
    return pd.DataFrame(rows, index=series.index, columns=COLUMNS)

