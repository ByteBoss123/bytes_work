"""Medication extraction from real clinical notes, scored against clinician annotations.

Data: OHSU Glaucoma Medication Dataset (Chen et al., TVST 2022; BSD-3-Clause): 480 de-identified
ophthalmology visit notes from Casey Eye Institute with sentence-level gold labels for drug,
frequency, route and other medication attributes. https://github.com/jche253/Glaucoma_Med_Dataset

Systems compared on the same held-out notes:
  1. Lexicon baseline: drug/frequency/route strings learned from the training notes only.
  2. LLM extractor: an instruction-tuned open model prompted with few-shot training examples,
     returning JSON that is mapped back onto the sentence text (strings not found = hallucinations).

Then each annotated drug name is mapped to RxNorm (NLM RxNav) and joined to the real CY2026 CMS
Part D formulary to show how often plans require prior authorization or step therapy for it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = 'https://github.com/jche253/Glaucoma_Med_Dataset'
DATA_COMMIT = '15e67ea86ed0678425ec0948032f3fe9f77f30db'
TYPES = ['DRUG', 'FREQUENCY', 'ROUTE']
TRIM = ' \t\n\r\xa0.,;:-()[]'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- data

def load(path):
    """Parse the annotation JSONL. De-identification masked some record ids ("id": *****),
    which makes those lines invalid JSON; the id is restored as null. Lines still unparseable
    are counted and dropped."""
    rows, bad = [], 0
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        line = re.sub(r'^\{"id": \*+,', '{"id": null,', line)
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        spans = []
        for s, e, t in r['labels']:
            s, e = trim_span(r['text'], s, e)
            if e > s:
                spans.append((s, e, t))
        rows.append({'text': r['text'], 'spans': spans})
    return rows, bad


def trim_span(text, s, e):
    s, e = max(0, min(s, len(text))), max(0, min(e, len(text)))
    while s < e and text[s] in TRIM:
        s += 1
    while e > s and text[e - 1] in TRIM:
        e -= 1
    return s, e


def masked(row):
    """True if de-identification masked any evaluated gold entity (its text is asterisks)."""
    return any('*' in row['text'][s:e] for s, e, t in row['spans'] if t in TYPES)


def split(rows, test_fraction=0.2):
    """Annotations are stored note by note and each note's text begins with an 'HPI:' line.
    Cut at the first note start after the 80% mark so no note straddles train and test."""
    starts = [i for i, r in enumerate(rows) if r['text'].startswith('HPI:')]
    target = int(len(rows) * (1 - test_fraction))
    cut = next((i for i in starts if i >= target), None)
    if cut is None:
        raise ValueError('no note boundary after the split point')
    start_set = set(starts)
    block, b = [], -1
    for i in range(len(rows)):
        if i in start_set:
            b += 1
        block.append(max(b, 0))
    return rows[:cut], rows[cut:], block[cut:], {'cut_index': cut, 'note_starts': len(starts)}


# ---------------------------------------------------------------- lexicon baseline

def norm(s):
    return re.sub(r'\s+', ' ', s.lower().replace('\xa0', ' ')).strip()


class Lexicon:
    """String lexicons learned from training gold spans. A string is kept for a type if, across
    all training sentences where it appears as a whole word, it was labeled with that type at
    least `min_precision` of the time (filters strings like 'od' that are usually exam text)."""

    def __init__(self, train, min_precision=0.5, min_count=1):
        self.entries = {}
        cands = {t: {} for t in TYPES}
        for r in train:
            for s, e, t in r['spans']:
                if t in TYPES:
                    k = norm(r['text'][s:e])
                    if len(k) >= 2 and '*' not in k:
                        cands[t][k] = cands[t].get(k, 0) + 1
        stats = {}
        for t in TYPES:
            keep = []
            for k, n in cands[t].items():
                pat = self.pattern([k])
                occ = sum(len(pat.findall(r['text'])) for r in train)
                prec = n / occ if occ else 0.0
                if n >= min_count and prec >= min_precision:
                    keep.append(k)
            self.entries[t] = sorted(keep, key=len, reverse=True)
            stats[t] = {'candidates': len(cands[t]), 'kept': len(keep)}
        self.stats = stats
        self.patterns = {t: self.pattern(v) for t, v in self.entries.items() if v}

    @staticmethod
    def pattern(strings):
        alts = '|'.join(re.escape(s).replace(r'\ ', r'\s+') for s in sorted(strings, key=len, reverse=True))
        return re.compile(r'(?<![\w])(?:' + alts + r')(?![\w])', re.IGNORECASE)

    def predict(self, text):
        out = []
        for t, pat in self.patterns.items():
            for m in pat.finditer(text):
                out.append((m.start(), m.end(), t))
        return out


# ---------------------------------------------------------------- LLM extractor

SYSTEM = ("You extract ophthalmic medication mentions from one sentence of an eye-clinic note. "
          "Include eye drops, eye ointments, artificial tears and eye-related oral drugs such as acetazolamide. "
          "Copy text exactly as written in the sentence. For each medication give drug, frequency and route; "
          "use null when absent. Routes are things like OU, OD, OS, both eyes, PO. "
          "If the sentence mentions no medication, return an empty list.")
SCHEMA = {'type': 'object', 'properties': {'medications': {'type': 'array', 'items': {
    'type': 'object', 'properties': {'drug': {'type': 'string'},
                                     'frequency': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
                                     'route': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}},
    'required': ['drug', 'frequency', 'route']}}}, 'required': ['medications']}


def few_shot(train, k_pos=5, k_neg=2):
    """Deterministic few-shot examples from the training notes only."""
    pos, neg, seen = [], [], set()
    for r in train:
        if masked(r):
            continue
        types = {t for _, _, t in r['spans'] if t in TYPES}
        drugs = [norm(r['text'][s:e]) for s, e, t in r['spans'] if t == 'DRUG']
        if types == set(TYPES) and drugs and drugs[0] not in seen and len(pos) < k_pos:
            seen.add(drugs[0])
            pos.append(r)
        elif not r['spans'] and len(r['text']) > 30 and len(neg) < k_neg and re.search(r'\bO[UDS]\b', r['text']):
            neg.append(r)
    return pos + neg


def gold_json(r):
    """Gold sentence annotation as the target JSON (pairs each drug with the attributes that follow it)."""
    meds, cur = [], None
    for s, e, t in sorted(r['spans']):
        if t == 'DRUG':
            cur = {'drug': r['text'][s:e], 'frequency': None, 'route': None}
            meds.append(cur)
        elif t in ('FREQUENCY', 'ROUTE') and cur is not None and cur[t.lower()] is None:
            cur[t.lower()] = r['text'][s:e]
    return {'medications': meds}


def messages(sentence, shots):
    m = [{'role': 'system', 'content': SYSTEM}]
    for r in shots:
        m.append({'role': 'user', 'content': r['text']})
        m.append({'role': 'assistant', 'content': json.dumps(gold_json(r))})
    m.append({'role': 'user', 'content': sentence})
    return m


def to_spans(text, parsed):
    """Map LLM strings onto the sentence. Returns spans and the strings that were not found."""
    spans, missing, used = [], [], []
    low = text.lower()
    for med in (parsed or {}).get('medications', []) or []:
        for key, t in (('drug', 'DRUG'), ('frequency', 'FREQUENCY'), ('route', 'ROUTE')):
            v = med.get(key) if isinstance(med, dict) else None
            if not v or not isinstance(v, str) or not v.strip(' .,'):
                continue
            v = v.strip()
            start, found = 0, None
            while True:
                i = low.find(v.lower(), start)
                if i < 0:
                    break
                s, e = trim_span(text, i, i + len(v))
                if (s, e, t) not in used:
                    found = (s, e, t)
                    break
                start = i + 1
            if found:
                used.append(found)
                spans.append(found)
            else:
                missing.append({'type': t, 'text': v})
    return spans, missing


class LlamaExtractor:
    def __init__(self, model_path, shots, threads=None):
        from llama_cpp import Llama
        self.llm = Llama(model_path=str(model_path), n_ctx=4096, n_threads=threads or os.cpu_count(),
                         seed=0, verbose=False)
        self.shots = shots

    def __call__(self, sentence):
        out = self.llm.create_chat_completion(
            messages=messages(sentence, self.shots), temperature=0.0, max_tokens=200,
            response_format={'type': 'json_object', 'schema': SCHEMA})
        raw = out['choices'][0]['message']['content']
        try:
            return json.loads(raw), raw
        except json.JSONDecodeError:
            return None, raw


# ---------------------------------------------------------------- metrics

def score(gold_rows, pred_spans, types=TYPES):
    """Micro precision/recall/F1 per type, strict (exact offsets) and lenient (any overlap)."""
    res = {}
    for t in types + ['ALL']:
        tt = types if t == 'ALL' else [t]
        tp = fp = fn = ltp = 0
        for r, p in zip(gold_rows, pred_spans):
            g = {(s, e, x) for s, e, x in r['spans'] if x in tt}
            q = {(s, e, x) for s, e, x in p if x in tt}
            tp += len(g & q)
            fp += len(q - g)
            fn += len(g - q)
            ltp += sum(1 for (s, e, x) in g if any(x == y and s < f and b < e for b, f, y in q))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        lrec = ltp / (tp + fn) if tp + fn else 0.0
        res[t] = {'gold': tp + fn, 'predicted': tp + fp, 'tp': tp, 'precision': prec, 'recall': rec,
                  'f1': 2 * prec * rec / (prec + rec) if prec + rec else 0.0, 'lenient_recall': lrec}
    return res


def counts(gold_rows, pred_spans, t):
    out = []
    for r, p in zip(gold_rows, pred_spans):
        g = {(s, e) for s, e, x in r['spans'] if x == t}
        q = {(s, e) for s, e, x in p if x == t}
        out.append((len(g & q), len(q - g), len(g - q)))
    return np.array(out)


def f1(c):
    tp, fp, fn = c.sum(0)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def bootstrap_diff(gold_rows, a, b, blocks, reps=1000, seed=0):
    """F1(b) - F1(a) per type, resampling whole notes (blocks) with replacement."""
    rng = np.random.default_rng(seed)
    blocks = np.asarray(blocks)
    uniq = np.unique(blocks)
    idx = [np.flatnonzero(blocks == u) for u in uniq]
    out = {}
    for t in TYPES + ['ALL']:
        ca = sum(counts(gold_rows, a, x) for x in (TYPES if t == 'ALL' else [t]))
        cb = sum(counts(gold_rows, b, x) for x in (TYPES if t == 'ALL' else [t]))
        d = []
        for _ in range(reps):
            ix = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
            d.append(f1(cb[ix]) - f1(ca[ix]))
        out[t] = {'point': f1(cb) - f1(ca), 'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}
    return out


def top_errors(gold_rows, pred, t, k=10):
    fp, fn = {}, {}
    for r, p in zip(gold_rows, pred):
        g = {(s, e) for s, e, x in r['spans'] if x == t}
        q = {(s, e) for s, e, x in p if x == t}
        for s, e in q - g:
            fp[norm(r['text'][s:e])] = fp.get(norm(r['text'][s:e]), 0) + 1
        for s, e in g - q:
            fn[norm(r['text'][s:e])] = fn.get(norm(r['text'][s:e]), 0) + 1
    srt = lambda d: sorted(d.items(), key=lambda kv: -kv[1])[:k]  # noqa: E731
    return {'false_positives': srt(fp), 'false_negatives': srt(fn)}


# ---------------------------------------------------------------- RxNorm + formulary link

def rxnav(path, **params):
    url = 'https://rxnav.nlm.nih.gov/REST/' + path + ('?' + urllib.parse.urlencode(params) if params else '')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'pbm-portfolio'}), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(1 + attempt)
    return {}


def rx_concepts(name):
    """Exact RxNorm name match first; otherwise the top approximate match (recorded as such)."""
    ids = rxnav('rxcui.json', name=name, search=2).get('idGroup', {}).get('rxnormId', [])
    method = 'exact_or_normalized'
    if not ids:
        cand = rxnav('approximateTerm.json', term=name, maxEntries=1).get('approximateGroup', {}).get('candidate', [])
        ids = [cand[0]['rxcui']] if cand else []
        method = 'approximate' if ids else 'unmapped'
    products = set()
    for i in ids[:3]:
        rel = rxnav(f'rxcui/{i}/related.json', tty='SCD SBD GPCK BPCK')  # space-separated; urlencode turns spaces into '+'
        for g in rel.get('relatedGroup', {}).get('conceptGroup', []) or []:
            for c in g.get('conceptProperties', []) or []:
                products.add(c['rxcui'])
    return method, sorted(ids), sorted(products)


def formulary_link(rows, raw_dir, top=25):
    from pbm import formulary as F
    mentions = {}
    for r in rows:
        for s, e, t in r['spans']:
            if t == 'DRUG' and '*' not in r['text'][s:e]:
                k = norm(r['text'][s:e])
                mentions[k] = mentions.get(k, 0) + 1
    form, plans, src = F.load_year(2026, Path(raw_dir))
    F.validate_formulary(form, 2026)
    pairs = F.pairs(form)
    n_forms = pairs.formulary_id.nunique()
    by_rx = pairs.groupby('rxcui')
    out = []
    for name, n in sorted(mentions.items(), key=lambda kv: -kv[1])[:top]:
        method, ids, products = rx_concepts(name)
        sub = pairs[pairs.rxcui.isin(products)]
        if len(sub):
            f = sub.groupby('formulary_id').agg(pa=('pa', 'max'), st=('st', 'max'), ql=('ql', 'max'))
            row = {'formularies_listing': int(len(f)), 'share_of_formularies': len(f) / n_forms,
                   'pa_share': float(f.pa.mean()), 'step_therapy_share': float(f.st.mean()),
                   'quantity_limit_share': float(f.ql.mean())}
        else:
            row = {'formularies_listing': 0, 'share_of_formularies': 0.0, 'pa_share': None,
                   'step_therapy_share': None, 'quantity_limit_share': None}
        out.append({'note_term': name, 'mentions': n, 'rxnorm_match': method, 'rxnorm_ids': ids,
                    'product_rxcuis': len(products), **row})
        log('link', name, method, row)
    del by_rx
    return out, {'formularies_2026': int(n_forms), 'source_sha256': src['sha256']}


# ---------------------------------------------------------------- run

def run(data_dir, out, model_path=None, raw_dir=None, limit=None, boot=1000, extractor=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    ann = data_dir / 'Annotations' / 'Combined_annotations.jsonl'
    rows, bad = load(ann)
    train, test, blocks, split_info = split(rows)
    test_eval = [(r, b) for r, b in zip(test, blocks) if not masked(r)]
    masked_test = len(test) - len(test_eval)
    if limit:
        test_eval = test_eval[:limit]
    test_rows = [r for r, _ in test_eval]
    test_blocks = [b for _, b in test_eval]
    log('rows', len(rows), 'unparseable', bad, 'train', len(train), 'test', len(test_rows), 'masked test excluded', masked_test)

    lex = Lexicon([r for r in train if not masked(r)])
    base = [lex.predict(r['text']) for r in test_rows]
    results = {'lexicon_baseline': score(test_rows, base)}
    # Ceiling of the JSON output format: gold labels pushed through the same JSON -> span mapping.
    oracle = [to_spans(r['text'], gold_json(r))[0] for r in test_rows]
    results['json_format_ceiling'] = score(test_rows, oracle)
    log('baseline', json.dumps(results['lexicon_baseline']['ALL']))

    shots = few_shot(train)
    llm_meta = None
    if extractor is None and model_path:
        extractor = LlamaExtractor(model_path, shots)
    if extractor is not None:
        llm, missing, raws, invalid, times = [], [], [], 0, []
        for i, r in enumerate(test_rows):
            t0 = time.perf_counter()
            parsed, raw = extractor(r['text'])
            times.append(time.perf_counter() - t0)
            if parsed is None:
                invalid += 1
            sp, miss = to_spans(r['text'], parsed)
            llm.append(sp)
            missing.append(miss)
            raws.append(raw)
            if i % 100 == 0:
                log('llm', i, len(test_rows), f'{np.mean(times):.2f}s/sentence')
        results['llm'] = score(test_rows, llm)
        n_strings = sum(len(s) for s in llm) + sum(len(m) for m in missing)
        n_missing = sum(len(m) for m in missing)
        llm_meta = {'model_file': Path(model_path).name if model_path else 'injected',
                    'model_sha256': hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if model_path else None,
                    'few_shot_examples': len(shots), 'decoding': 'temperature 0, JSON-schema constrained',
                    'invalid_json': invalid, 'strings_returned': n_strings,
                    'strings_not_in_sentence': n_missing,
                    'not_in_sentence_rate': n_missing / n_strings if n_strings else 0.0,
                    'seconds_per_sentence_mean': float(np.mean(times)),
                    'seconds_per_sentence_p95': float(np.quantile(times, .95))}
        results['llm_minus_baseline_f1'] = bootstrap_diff(test_rows, base, llm, test_blocks, reps=boot)
        errors = {t: {'baseline': top_errors(test_rows, base, t), 'llm': top_errors(test_rows, llm, t)} for t in TYPES}
        with open(out / 'notes_test_predictions.jsonl', 'w') as f:
            for r, b, l, m, raw in zip(test_rows, base, llm, missing, raws):
                f.write(json.dumps({'text': r['text'], 'gold': r['spans'], 'baseline': b, 'llm': l,
                                    'llm_not_in_sentence': m, 'llm_raw': raw}) + '\n')
    else:
        errors = {t: {'baseline': top_errors(test_rows, base, t)} for t in TYPES}

    link = link_meta = None
    if raw_dir:
        link, link_meta = formulary_link(rows, raw_dir)

    gold_counts = {t: sum(1 for r in rows for s, e, x in r['spans'] if x == t) for t in TYPES}
    result = {
        'data': {'source': DATA_REPO, 'commit': DATA_COMMIT, 'license': 'BSD-3-Clause',
                 'description': '480 de-identified ophthalmology visit notes (OHSU Casey Eye Institute, 2019-2020) with clinician sentence-level medication labels',
                 'annotation_sha256': hashlib.sha256(ann.read_bytes()).hexdigest(),
                 'sentences_parsed': len(rows), 'lines_unparseable': bad, 'gold_entities_all': gold_counts},
        'split': {**split_info, 'train_sentences': len(train), 'test_sentences': len(test),
                  'test_sentences_masked_excluded': masked_test, 'test_sentences_evaluated': len(test_rows),
                  'test_notes': len(set(test_blocks)), 'rule': 'cut at first note start after 80% of sentences'},
        'evaluated_types': TYPES, 'lexicon': lex.stats, 'results': results, 'llm': llm_meta,
        'bootstrap_note': f'{boot} note-level bootstrap replicates of the F1 difference',
        'top_errors': errors, 'formulary_link': link, 'formulary_link_meta': link_meta,
        'limitations': ['Ophthalmology notes from one institution; not representative of all clinical text.',
                        'Gold labels cover ophthalmic medications; extracting other drugs counts as a false positive.',
                        'Some gold labels are inconsistent (e.g. qhs labeled DOSAGE); scores inherit that noise.',
                        'Strict scoring needs exact character offsets; lenient recall counts any overlap.']}
    (out / 'notes_extraction.json').write_text(json.dumps(result, indent=2, default=float))
    log(json.dumps({k: v for k, v in results.items()}, indent=1, default=float))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'reports')
    ap.add_argument('--model', type=Path, default=None)
    ap.add_argument('--raw', type=Path, default=None, help='CMS archive cache; enables the 2026 formulary link')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--link-only', action='store_true', help='recompute only the formulary link into an existing notes_extraction.json')
    a = ap.parse_args()
    if a.link_only:
        res_path = a.out / 'notes_extraction.json'
        res = json.loads(res_path.read_text())
        rows, _ = load(a.data / 'Annotations' / 'Combined_annotations.jsonl')
        res['formulary_link'], res['formulary_link_meta'] = formulary_link(rows, a.raw)
        res_path.write_text(json.dumps(res, indent=2, default=float))
    else:
        run(a.data, a.out, a.model, a.raw, a.limit, a.boot)
