"""Retrieval-augmented question answering over real FDA drug labels and CY2026 Part D coverage.

Drugs: the highest-spend generic names in the real CMS Part D prescriber extract
(reports/cms_spend_by_generic.csv). Corpus, all real public data:
  * FDA prescribing-information text for each drug (openFDA drug label API): boxed warning,
    indications, dosage and administration, dosage forms, contraindications, warnings.
  * One coverage summary per drug computed from the real CY2026 CMS Part D formulary file
    (share of listing formularies requiring prior authorization, step therapy, quantity limits).

Questions are generated from structured fields that are NOT in the retrieved text, so the
system must recover them from the unstructured corpus and nothing is hand-labeled:
  * route      -> openFDA `openfda.route` metadata (label text only describes administration)
  * boxed      -> whether the label has a boxed-warning section
  * pa_share   -> prior-authorization share from the formulary file (stated in the coverage doc)

Compared on the same questions: closed-book LLM vs RAG (BM25 + dense hybrid retrieval), with
retrieval hit rate, answer accuracy, abstention, citation precision, and a drug-level bootstrap.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = [('boxed_warning', 'Boxed warning'), ('indications_and_usage', 'Indications and usage'),
            ('dosage_and_administration', 'Dosage and administration'),
            ('dosage_forms_and_strengths', 'Dosage forms and strengths'),
            ('contraindications', 'Contraindications'), ('warnings_and_cautions', 'Warnings and precautions'),
            ('warnings', 'Warnings')]
SECTION_WORDS = 600
CHUNK_WORDS, OVERLAP = 180, 30
TOP_K = 5
ROUTES = [  # canonical openFDA route -> phrases that express it in an answer
    ('ORAL', r'\boral(ly)?\b|\bby mouth\b|\btablets? (taken|swallowed)\b'),
    ('SUBCUTANEOUS', r'\bsubcutaneous(ly)?\b|\bunder the skin\b'),
    ('INTRAVENOUS', r'\bintravenous(ly)?\b|\bi\.?v\.? (infusion|injection)\b'),
    ('INTRAMUSCULAR', r'\bintramuscular(ly)?\b'),
    ('OPHTHALMIC', r'\bophthalmic\b|\b(in|into) the eyes?\b'),
    ('RESPIRATORY (INHALATION)', r'\binhal(ed|ation|er)\b|\brespiratory\b'),
    ('TOPICAL', r'\btopical(ly)?\b'),
    ('TRANSDERMAL', r'\btransdermal\b'),
    ('NASAL', r'\b(intra)?nasal(ly)?\b'),
    ('SUBLINGUAL', r'\bsublingual(ly)?\b'),
    ('BUCCAL', r'\bbuccal\b'),
    ('RECTAL', r'\brectal(ly)?\b'),
    ('VAGINAL', r'\bvaginal(ly)?\b'),
    ('INTRAVITREAL', r'\bintravitreal\b'),
    ('INTRADERMAL', r'\bintradermal\b'),
    ('CUTANEOUS', r'\bcutaneous\b'),
]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def norm(s):
    return re.sub(r'\s+', ' ', str(s)).strip()


# ---------------------------------------------------------------- data

def select_generics(spend_csv, n):
    """Top-n generic names by recorded drug cost; CMS appends salt/form after a comma."""
    d = pd.read_csv(spend_csv).sort_values('drug_cost', ascending=False)
    out, seen = [], set()
    for name in d.generic_name:
        g = norm(str(name).split(',')[0]).lower()
        if g and g not in seen:
            seen.add(g)
            out.append(g)
        if len(out) >= n:
            break
    return out


def openfda_search(generic):
    q = f'openfda.generic_name:"{generic}" AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
    url = 'https://api.fda.gov/drug/label.json?search=' + urllib.parse.quote(q, safe=':"').replace('%20', '+') + '&limit=25'
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'pbm-portfolio'}), timeout=60) as r:
                return json.load(r).get('results', [])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            time.sleep(2 + 2 * attempt)
        except Exception:
            time.sleep(2 + 2 * attempt)
    return []


def pick_label(generic, results):
    """Prefer a branded single-ingredient label whose generic name matches exactly; newest first."""
    def ok(r):
        o = r.get('openfda', {})
        gens = [g.lower() for g in o.get('generic_name', [])]
        return o.get('route') and o.get('brand_name') and generic in gens and any(r.get(k) for k, _ in SECTIONS[1:3])
    cands = [r for r in results if ok(r)]
    if not cands:
        return None
    branded = [r for r in cands if r['openfda']['brand_name'][0].lower() != generic]
    pool = branded or cands
    return max(pool, key=lambda r: r.get('effective_time', ''))


def fetch_labels(generics, cache):
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    labels, misses = [], []
    for g in generics:
        r = pick_label(g, openfda_search(g))
        time.sleep(0.3)
        if r is None:
            misses.append(g)
            continue
        o = r['openfda']
        labels.append({'generic': g, 'brand': norm(o['brand_name'][0]), 'set_id': r.get('set_id'),
                       'effective_time': r.get('effective_time'), 'routes': sorted(set(o['route'])),
                       'boxed': bool(r.get('boxed_warning')),
                       'sections': {k: norm(' '.join(r[k])) for k, _ in SECTIONS if r.get(k)}})
    out = {'labels': labels, 'not_found': misses}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def dedupe(labels):
    seen, out = set(), []
    for l in labels:
        k = l['brand'].lower()
        if k not in seen and l['set_id'] not in seen:
            seen.update([k, l['set_id']])
            out.append(l)
    return out


def coverage(generics, raw_dir):
    """Share of CY2026 formularies listing each drug that apply PA / step therapy / QL."""
    from pbm import formulary as F
    from pbm.notes_extraction import rx_concepts
    form, plans, src = F.load_year(2026, Path(raw_dir))
    F.validate_formulary(form, 2026)
    pairs = F.pairs(form)
    n_forms = int(pairs.formulary_id.nunique())
    out = {}
    for g in generics:
        method, ids, products = rx_concepts(g)
        sub = pairs[pairs.rxcui.isin(products)]
        if not len(sub):
            continue
        f = sub.groupby('formulary_id').agg(pa=('pa', 'max'), st=('st', 'max'), ql=('ql', 'max'))
        out[g] = {'listing': int(len(f)), 'formularies': n_forms, 'pa_pct': round(100 * float(f.pa.mean()), 1),
                  'st_pct': round(100 * float(f.st.mean()), 1), 'ql_pct': round(100 * float(f.ql.mean()), 1),
                  'rxnorm_match': method}
    return out, {'formularies_2026': n_forms, 'source_sha256': src['sha256']}


# ---------------------------------------------------------------- corpus and questions

def chunks_for(label):
    out = []
    for key, title in SECTIONS:
        text = label['sections'].get(key)
        if not text:
            continue
        words = text.split()[:SECTION_WORDS]
        step = CHUNK_WORDS - OVERLAP
        for i in range(0, max(1, len(words) - OVERLAP), step):
            body = ' '.join(words[i:i + CHUNK_WORDS])
            if body:
                out.append({'doc': 'label:' + label['generic'], 'drug': label['generic'], 'section': key,
                            'text': f"{label['brand']} ({label['generic']}) prescribing information, {title}: {body}"})
    return out


def coverage_doc(generic, brand, c):
    return {'doc': 'coverage:' + generic, 'drug': generic, 'section': 'coverage',
            'text': (f"Medicare Part D contract year 2026 formulary coverage for {generic} ({brand}): listed on "
                     f"{c['listing']} of {c['formularies']} formularies. Among formularies that list it, prior "
                     f"authorization is required on {c['pa_pct']:.1f}%, step therapy on {c['st_pct']:.1f}% and "
                     f"quantity limits on {c['ql_pct']:.1f}%.")}


def build(labels, cov):
    corpus, questions = [], []
    for l in labels:
        corpus += chunks_for(l)
        questions.append({'type': 'route', 'drug': l['generic'], 'doc': 'label:' + l['generic'],
                          'question': f"By what route is {l['brand']} administered?", 'gold': l['routes']})
        questions.append({'type': 'boxed', 'drug': l['generic'], 'doc': 'label:' + l['generic'],
                          'question': f"Does the {l['brand']} prescribing information carry a boxed warning?",
                          'gold': l['boxed']})
        if l['generic'] in cov:
            corpus.append(coverage_doc(l['generic'], l['brand'], cov[l['generic']]))
            questions.append({'type': 'pa_share', 'drug': l['generic'], 'doc': 'coverage:' + l['generic'],
                              'question': (f"Among 2026 Medicare Part D formularies that list {l['generic']}, "
                                           "what percentage require prior authorization?"),
                              'gold': cov[l['generic']]['pa_pct']})
    for i, c in enumerate(corpus):
        c['id'] = f'C{i}'
    return corpus, questions


def misspell(name):
    """Deterministic typo: swap the 3rd and 4th letters of the first word (e.g. Eliquis -> Elqiuis)."""
    w = name.split()
    if len(w[0]) < 5:
        return None
    x = w[0]
    w[0] = x[:2] + x[3] + x[2] + x[4:]
    return ' '.join(w) if ' '.join(w).lower() != name.lower() else None


def variants(labels, qs):
    """Robustness set built from the same real gold answers: misspelled brand, and generic name
    in place of brand, for every route and boxed-warning question."""
    by = {l['generic']: l for l in labels}
    out = []
    for q in qs:
        if q['type'] not in ('route', 'boxed'):
            continue
        brand = by[q['drug']]['brand']
        typo = misspell(brand)
        if typo:
            out.append({**q, 'variant': 'misspelled_brand', 'question': q['question'].replace(brand, typo)})
        if brand.lower() != q['drug']:
            out.append({**q, 'variant': 'generic_name', 'question': q['question'].replace(brand, q['drug'])})
    return out


# ---------------------------------------------------------------- retrieval

def tokens(s):
    return re.findall(r'[a-z0-9]+', s.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [tokens(t) for t in texts]
        self.k1, self.b = k1, b
        self.avg = np.mean([len(d) for d in self.docs])
        self.df = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(self.docs)
        self.tf = [{t: d.count(t) for t in set(d)} for d in self.docs]

    def scores(self, q):
        out = np.zeros(self.n)
        for t in set(tokens(q)):
            if t not in self.df:
                continue
            idf = math.log(1 + (self.n - self.df[t] + .5) / (self.df[t] + .5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.avg))
        return out


class Retriever:
    """BM25, dense (injected embedding function), reciprocal-rank-fusion hybrid, and a hybrid
    restricted to the drugs named in the question (entity filter on brand or generic name)."""

    def __init__(self, corpus, embed=None, names=None):
        self.corpus = corpus
        # names: {lowercase brand or generic name: generic}; longest names matched first
        self.names = sorted((names or {}).items(), key=lambda kv: -len(kv[0]))
        self.bm25 = BM25([c['text'] for c in corpus])
        self.embed = embed
        self.emb = embed([c['text'] for c in corpus], query=False) if embed else None

    def rank(self, q, method):
        if method == 'bm25':
            return list(np.argsort(-self.bm25.scores(q), kind='stable'))
        if method == 'dense':
            return list(np.argsort(-(self.emb @ self.embed([q], query=True)[0]), kind='stable'))
        b, d = self.rank(q, 'bm25')[:50], self.rank(q, 'dense')[:50]
        rrf = {}
        for ranking in (b, d):
            for r, i in enumerate(ranking):
                rrf[i] = rrf.get(i, 0) + 1 / (60 + r + 1)
        return sorted(rrf, key=lambda i: (-rrf[i], i))

    def drugs_in(self, q):
        """Drugs named in the question: exact brand/generic match, then fuzzy match for misspellings."""
        t, found = q.lower(), set()
        for name, generic in self.names:
            if re.search(r'(?<![a-z0-9])' + re.escape(name) + r'(?![a-z0-9])', t):
                found.add(generic)
        if found:
            return found, 'exact'
        words = tokens(q)
        grams = {' '.join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
        best = {}
        for g in grams:
            if len(g) < 5:
                continue
            for name, generic in self.names:
                if abs(len(name) - len(g)) > 2:
                    continue
                r = difflib.SequenceMatcher(None, g, name).ratio()
                if r >= 0.85 and r > best.get(generic, 0):
                    best[generic] = r
        if best:
            top = max(best.values())
            return {d for d, r in best.items() if r == top}, 'fuzzy'
        return set(), 'none'

    def infer_drug(self, q, depth=10, share=0.5):
        """No name recognised: if one drug dominates the top hybrid results, restrict to it."""
        ranked = self.rank(q, 'hybrid')[:depth]
        w = {}
        for r, i in enumerate(ranked):
            d = self.corpus[i].get('drug')
            w[d] = w.get(d, 0) + 1 / (r + 1)
        if not w:
            return set()
        d, v = max(w.items(), key=lambda kv: kv[1])
        return {d} if v / sum(w.values()) >= share else set()

    def resolve(self, q):
        drugs, how = self.drugs_in(q)
        if not drugs:
            drugs = self.infer_drug(q)
            how = 'inferred' if drugs else 'unfiltered'
        return drugs, how

    def top(self, q, method, k=TOP_K):
        if method == 'hybrid_drug_filter':
            drugs, _ = self.resolve(q)
            ranked = self.rank(q, 'hybrid')
            if drugs:
                ranked = [i for i in ranked if self.corpus[i].get('drug') in drugs]
                if len(ranked) < k:  # RRF lists cover only the top 50 of each ranker
                    seen = set(ranked)
                    ranked += [i for i in range(len(self.corpus)) if self.corpus[i].get('drug') in drugs and i not in seen]
            return [self.corpus[i] for i in ranked[:k]]
        return [self.corpus[i] for i in self.rank(q, method)[:k]]


def st_embedder(model_name='BAAI/bge-small-en-v1.5'):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device='cpu')
    prefix = 'Represent this sentence for searching relevant passages: '

    def embed(texts, query=False):
        texts = [prefix + t for t in texts] if query else texts
        return m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return embed


# ---------------------------------------------------------------- generation

SCHEMA = {'type': 'object', 'properties': {
    'answer': {'type': 'string'}, 'citations': {'type': 'array', 'items': {'type': 'string'}},
    'abstain': {'type': 'boolean'}}, 'required': ['answer', 'citations', 'abstain']}
CLOSED = ('You answer questions about US prescription drugs and Medicare Part D coverage from your own knowledge. '
          'Answer in one short sentence. Return JSON with answer, citations (empty list) and abstain '
          '(true only if you cannot answer).')
OPEN = ('You answer questions about US prescription drugs and Medicare Part D coverage using ONLY the numbered '
        'context passages. Answer in one short sentence. Cite the ids of the passages you used, for example ["C12"]. '
        'If the passages do not contain the answer, set abstain to true. For a boxed-warning question, answer no '
        'if the passages for that drug contain no boxed-warning section.')


def messages(q, ctx=None):
    if ctx is None:
        return [{'role': 'system', 'content': CLOSED}, {'role': 'user', 'content': q}]
    body = '\n\n'.join(f"[{c['id']}] {c['text']}" for c in ctx)
    return [{'role': 'system', 'content': OPEN}, {'role': 'user', 'content': f'Context:\n{body}\n\nQuestion: {q}'}]


class LlamaQA:
    def __init__(self, model_path):
        from llama_cpp import Llama
        self.llm = Llama(model_path=str(model_path), n_ctx=4096, n_batch=512, n_threads=os.cpu_count(),
                         seed=0, verbose=False)

    def __call__(self, msgs):
        out = self.llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=120,
                                              response_format={'type': 'json_object', 'schema': SCHEMA})
        raw = out['choices'][0]['message']['content']
        try:
            return json.loads(raw), raw
        except json.JSONDecodeError:
            return None, raw


# ---------------------------------------------------------------- scoring

def parse_routes(text):
    t = str(text).lower()
    return {name for name, pat in ROUTES if re.search(pat, t)}


def parse_yes_no(text):
    t = str(text).strip().lower()
    if t.startswith('yes'):
        return True
    if t.startswith('no'):
        return False
    if re.search(r"\b(does not|doesn't|no boxed|not carry|without a boxed|none)\b", t):
        return False
    if re.search(r'\bboxed warning\b', t):
        return True
    return None


def parse_pct(text):
    t = str(text).lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*percent', t) or re.search(r'(\d+(?:\.\d+)?)', t)
    if m:
        return float(next(g for g in m.groups() if g is not None))
    if re.search(r'\b(none|no formular|zero)\b', t):
        return 0.0
    return None


def correct(q, pred):
    if pred is None or pred.get('abstain'):
        return False
    a = pred.get('answer', '')
    if q['type'] == 'route':
        got = parse_routes(a)
        return bool(got) and got <= set(q['gold'])
    if q['type'] == 'boxed':
        return parse_yes_no(a) == q['gold']
    v = parse_pct(a)
    return v is not None and abs(v - q['gold']) <= 0.5


def drug_bootstrap(drugs, a, b, reps=2000, seed=0):
    """Accuracy(b) - accuracy(a), resampling drugs (all questions of a drug move together)."""
    rng = np.random.default_rng(seed)
    drugs = np.asarray(drugs)
    uniq = np.unique(drugs)
    idx = [np.flatnonzero(drugs == u) for u in uniq]
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = []
    for _ in range(reps):
        ix = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
        d.append(b[ix].mean() - a[ix].mean())
    return {'point': float(b.mean() - a.mean()), 'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}


def boxed_ablation(rows):
    if not rows:
        return None
    no = [r for r in rows if not r['gold']]
    out = {'questions': len(rows), 'drugs_without_boxed_warning': len(no)}
    for m in ('filtered', 'unfiltered'):
        out[m] = {'accuracy': float(np.mean([r[f'{m}_ok'] for r in rows])),
                  'false_yes_on_no_boxed_drugs': int(sum(r[f'{m}_said_yes'] for r in no))}
    out['filtered_minus_unfiltered'] = drug_bootstrap([r['drug'] for r in rows], [r['unfiltered_ok'] for r in rows],
                                                      [r['filtered_ok'] for r in rows])
    return out


def summarize(qs, ok, preds=None):
    out = {}
    for t in ['route', 'boxed', 'pa_share', 'ALL']:
        m = [i for i, q in enumerate(qs) if t == 'ALL' or q['type'] == t]
        if not m:
            continue
        row = {'n': len(m), 'accuracy': float(np.mean([ok[i] for i in m]))}
        if preds is not None:
            row['abstained'] = int(sum(1 for i in m if preds[i] and preds[i].get('abstain')))
            row['invalid_json'] = int(sum(1 for i in m if preds[i] is None))
        out[t] = row
    return out


# ---------------------------------------------------------------- run

def run(out, spend_csv, n_drugs=80, raw_dir=None, model_path=None, labels_cache=None,
        labels=None, cov=None, qa=None, embed=None, boot=2000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    generics = select_generics(spend_csv, n_drugs)
    fetched = {'labels': labels, 'not_found': []} if labels is not None else fetch_labels(generics, labels_cache or out / 'labels_cache.json')
    labels = dedupe(fetched['labels'])
    log('drugs', len(generics), 'labels', len(labels), 'not found', len(fetched['not_found']))
    cov_meta = None
    if cov is None:
        cov, cov_meta = coverage([l['generic'] for l in labels], raw_dir) if raw_dir else ({}, None)
    corpus, qs = build(labels, cov)
    log('corpus chunks', len(corpus), 'questions', len(qs))

    if embed is None:
        embed = st_embedder()
    names = {}
    for l in labels:
        names[l['generic'].lower()] = l['generic']
        names[l['brand'].lower()] = l['generic']
    ret = Retriever(corpus, embed, names)
    retrieval = {}
    for method in ['bm25', 'dense', 'hybrid', 'hybrid_drug_filter']:
        hits = [any(c['doc'] == q['doc'] for c in ret.top(q['question'], method)) for q in qs]
        retrieval[method] = summarize(qs, hits)
    boxed_q = [q for q in qs if q['type'] == 'boxed' and q['gold']]
    for method in ['hybrid', 'hybrid_drug_filter']:
        retrieval[f'boxed_section_hit_{method}'] = float(np.mean(
            [any(c['doc'] == q['doc'] and c['section'] == 'boxed_warning' for c in ret.top(q['question'], method)) for q in boxed_q])) if boxed_q else None
    no_box = [q for q in qs if q['type'] == 'boxed' and not q['gold']]
    for method in ['hybrid', 'hybrid_drug_filter']:
        retrieval[f'other_drug_boxed_passage_in_top5_{method}'] = float(np.mean(
            [any(c['section'] == 'boxed_warning' and c['doc'] != q['doc'] for c in ret.top(q['question'], method)) for q in no_box])) if no_box else None
    var = variants(labels, qs)
    robustness = {'questions': len(var), 'construction': 'misspelled brand (3rd/4th letters swapped) and generic name in place of brand'}
    for v in ('misspelled_brand', 'generic_name'):
        vq = [q for q in var if q['variant'] == v]
        if not vq:
            continue
        row = {'n': len(vq), 'resolution': {}}
        for q in vq:
            how = ret.resolve(q['question'])[1]
            row['resolution'][how] = row['resolution'].get(how, 0) + 1
        for method in ('hybrid', 'hybrid_drug_filter'):
            row[f'hit_at_5_{method}'] = float(np.mean([any(c['doc'] == q['doc'] for c in ret.top(q['question'], method)) for q in vq]))
            nb = [q for q in vq if q['type'] == 'boxed' and not q['gold']]
            row[f'other_drug_boxed_in_top5_{method}'] = float(np.mean(
                [any(c['section'] == 'boxed_warning' and c['doc'] != q['doc'] for c in ret.top(q['question'], method)) for q in nb])) if nb else None
        robustness[v] = row
    log('retrieval', json.dumps(retrieval))
    log('robustness', json.dumps(robustness))

    if qa is None:
        qa = LlamaQA(model_path)
    rows, closed_ok, rag_ok, closed_p, rag_p, cite, ablation = [], [], [], [], [], [], []
    t0 = time.time()
    for i, q in enumerate(qs):
        cp, craw = qa(messages(q['question']))
        ctx = ret.top(q['question'], 'hybrid_drug_filter')
        rp, rraw = qa(messages(q['question'], ctx))
        if q['type'] == 'boxed':  # ablation: same question with unfiltered hybrid retrieval
            up, _ = qa(messages(q['question'], ret.top(q['question'], 'hybrid')))
            ablation.append({'gold': q['gold'], 'drug': q['drug'], 'filtered_ok': correct(q, rp), 'unfiltered_ok': correct(q, up),
                             'filtered_said_yes': parse_yes_no((rp or {}).get('answer', '')) is True,
                             'unfiltered_said_yes': parse_yes_no((up or {}).get('answer', '')) is True,
                             'unfiltered_answer': up})
        closed_p.append(cp)
        rag_p.append(rp)
        closed_ok.append(correct(q, cp))
        rag_ok.append(correct(q, rp))
        ids = {c['id']: c for c in ctx}
        cited = [ids[c] for c in (rp or {}).get('citations', []) if c in ids]
        cite.append({'cited': len(cited), 'unknown_ids': len([c for c in (rp or {}).get('citations', []) if c not in ids]),
                     'all_from_gold_doc': bool(cited) and all(c['doc'] == q['doc'] for c in cited)})
        rows.append({**q, 'context_ids': [c['id'] for c in ctx], 'context_docs': [c['doc'] for c in ctx],
                     'closed_book': cp, 'closed_raw': craw, 'closed_correct': closed_ok[-1],
                     'rag': rp, 'rag_raw': rraw, 'rag_correct': rag_ok[-1]})
        if i % 20 == 0:
            log('qa', i, len(qs), f'{(time.time() - t0) / (i + 1):.1f}s/question (both modes)')

    typo_boxed = [q for q in var if q['variant'] == 'misspelled_brand' and q['type'] == 'boxed']
    tb = []
    for q in typo_boxed:
        fp, _ = qa(messages(q['question'], ret.top(q['question'], 'hybrid_drug_filter')))
        up, _ = qa(messages(q['question'], ret.top(q['question'], 'hybrid')))
        tb.append({'gold': q['gold'], 'drug': q['drug'], 'filtered_ok': correct(q, fp), 'unfiltered_ok': correct(q, up),
                   'filtered_said_yes': parse_yes_no((fp or {}).get('answer', '')) is True,
                   'unfiltered_said_yes': parse_yes_no((up or {}).get('answer', '')) is True, 'unfiltered_answer': up})
    robustness['misspelled_boxed_answers'] = boxed_ablation(tb)
    log('robustness answers', json.dumps(robustness['misspelled_boxed_answers']))
    answered = [c for c, p in zip(cite, rag_p) if p and not p.get('abstain')]
    citation = {'rag_answers': len(answered),
                'with_valid_citation': int(sum(1 for c in answered if c['cited'])),
                'citation_precision': float(np.mean([c['all_from_gold_doc'] for c in answered if c['cited']])) if any(c['cited'] for c in answered) else None,
                'answers_citing_unknown_ids': int(sum(1 for c in answered if c['unknown_ids']))}
    drugs = [q['drug'] for q in qs]
    result = {
        'data': {'drug_selection': 'top generic names by recorded drug cost in reports/cms_spend_by_generic.csv (real CMS Part D prescriber extract)',
                 'drugs_requested': len(generics), 'labels_found': len(labels), 'labels_not_found': fetched['not_found'],
                 'label_source': 'openFDA drug label API (FDA SPL prescribing information)',
                 'labels_sha256': hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest(),
                 'coverage_source': 'CMS CY2026 Part D formulary file via RxNorm (NLM RxNav)', 'coverage_meta': cov_meta,
                 'drugs_with_coverage': len(cov), 'corpus_chunks': len(corpus), 'questions': len(qs),
                 'section_words_cap': SECTION_WORDS, 'chunk_words': CHUNK_WORDS, 'overlap': OVERLAP},
        'gold_answers': 'route = openFDA route metadata; boxed = presence of a boxed-warning section; pa_share = formulary-file share (+/-0.5 points)',
        'retrieval_hit_at_5': retrieval,
        'answer_accuracy': {'closed_book': summarize(qs, closed_ok, closed_p), 'rag_hybrid_drug_filter': summarize(qs, rag_ok, rag_p)},
        'boxed_warning_ablation': boxed_ablation(ablation),
        'robustness': robustness,
        'rag_minus_closed_book': {t: drug_bootstrap([d for d, q in zip(drugs, qs) if t == 'ALL' or q['type'] == t],
                                                    [o for o, q in zip(closed_ok, qs) if t == 'ALL' or q['type'] == t],
                                                    [o for o, q in zip(rag_ok, qs) if t == 'ALL' or q['type'] == t], reps=boot)
                                  for t in ['route', 'boxed', 'pa_share', 'ALL'] if any(t == 'ALL' or q['type'] == t for q in qs)},
        'citations': citation,
        'base_rates': {'boxed_true_share': float(np.mean([q['gold'] for q in qs if q['type'] == 'boxed'])) if any(q['type'] == 'boxed' for q in qs) else None},
        'generator': {'model_file': Path(model_path).name if model_path else 'injected',
                      'decoding': 'temperature 0, JSON-schema constrained', 'top_k': TOP_K,
                      'rag_retrieval': 'hybrid (BM25 + bge-small, RRF) restricted to drugs named in the question (exact, then fuzzy >= 0.85), else to a drug that dominates the top-10 hybrid results (>= 50% of rank weight), else unfiltered'},
        'limitations': ['Question templates are simple; they test retrieval and grounding, not clinical reasoning.',
                        'Boxed-warning "no" must be inferred from the absence of a boxed-warning passage.',
                        'Label sections are capped at 600 words each, so late details can be missing.',
                        'Coverage shares are computed from formulary rules, not claims or utilization.',
                        'The drug filter relies on the question naming or strongly implying one drug; misspellings are synthetic (letter swap).'],
    }
    (out / 'drug_rag.json').write_text(json.dumps(result, indent=2, default=float))
    with open(out / 'drug_rag_predictions.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + '\n')
    log(json.dumps({k: result[k] for k in ['retrieval_hit_at_5', 'answer_accuracy', 'rag_minus_closed_book', 'citations']}, indent=1))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'reports')
    ap.add_argument('--spend', type=Path, default=ROOT / 'reports' / 'cms_spend_by_generic.csv')
    ap.add_argument('--n', type=int, default=80)
    ap.add_argument('--raw', type=Path, default=None)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--labels-cache', type=Path, default=None)
    a = ap.parse_args()
    run(a.out, a.spend, a.n, a.raw, a.model, a.labels_cache)
