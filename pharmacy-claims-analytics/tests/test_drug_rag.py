"""Logic tests for the drug-label RAG pipeline on small hand-written fixtures (test inputs only)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pbm import drug_rag as R


def label(generic, brand, routes, boxed, text):
    s = {'indications_and_usage': f'{brand} is indicated for {text}.',
         'dosage_and_administration': f'Take {brand} {text}.'}
    if boxed:
        s['boxed_warning'] = f'WARNING: serious risk with {brand}.'
    return {'generic': generic, 'brand': brand, 'set_id': brand + '-id', 'effective_time': '20250101',
            'routes': routes, 'boxed': boxed, 'sections': s}


LABELS = [label('apixaban', 'Eliquis', ['ORAL'], True, 'orally twice daily to reduce stroke risk'),
          label('semaglutide', 'Ozempic', ['SUBCUTANEOUS'], True, 'subcutaneously once weekly for glycemic control'),
          label('latanoprost', 'Xalatan', ['OPHTHALMIC'], False, 'one drop in the affected eye in the evening')]
COV = {'apixaban': {'listing': 331, 'formularies': 331, 'pa_pct': 0.0, 'st_pct': 0.0, 'ql_pct': 12.4, 'rxnorm_match': 'x'},
       'semaglutide': {'listing': 300, 'formularies': 331, 'pa_pct': 87.3, 'st_pct': 1.0, 'ql_pct': 90.0, 'rxnorm_match': 'x'}}


def hash_embed(texts, query=False):
    """Deterministic bag-of-words hashing embedder for tests."""
    out = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        for w in R.tokens(t):
            out[i, hash(w) % 256] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


class Parsing(unittest.TestCase):
    def test_routes(self):
        self.assertEqual(R.parse_routes('It is taken orally (by mouth).'), {'ORAL'})
        self.assertEqual(R.parse_routes('Inject subcutaneously once weekly.'), {'SUBCUTANEOUS'})
        self.assertEqual(R.parse_routes('One drop into the eye.'), {'OPHTHALMIC'})

    def test_yes_no_and_pct(self):
        self.assertTrue(R.parse_yes_no('Yes, it carries a boxed warning.'))
        self.assertFalse(R.parse_yes_no('No, the label does not carry a boxed warning.'))
        self.assertEqual(R.parse_pct('About 87.3% of them require PA.'), 87.3)
        self.assertEqual(R.parse_pct('None of them do.'), 0.0)

    def test_correct_rules(self):
        q = {'type': 'route', 'gold': ['ORAL']}
        self.assertTrue(R.correct(q, {'answer': 'orally', 'abstain': False}))
        self.assertFalse(R.correct(q, {'answer': 'orally or intravenously', 'abstain': False}))
        self.assertFalse(R.correct(q, {'answer': 'orally', 'abstain': True}))
        self.assertTrue(R.correct({'type': 'pa_share', 'gold': 87.3}, {'answer': '87%', 'abstain': False}))
        self.assertFalse(R.correct({'type': 'pa_share', 'gold': 87.3}, {'answer': '80%', 'abstain': False}))


class Matching(unittest.TestCase):
    def test_clean_name_and_text_fallback(self):
        self.assertEqual(R.clean_name('cariprazine hcl'), 'cariprazine')
        self.assertEqual(R.clean_name('dapagliflozin propanediol'), 'dapagliflozin')
        self.assertEqual(R.clean_name('sacubitril/valsartan'), 'sacubitril/valsartan')
        f = R.text_fallback('Ozempic is administered subcutaneously ["C20"].')
        self.assertEqual((f['citations'], f['abstain'], f['format']), (['C20'], False, 'text_fallback'))
        self.assertTrue(R.correct({'type': 'route', 'gold': ['SUBCUTANEOUS']}, f))
        self.assertTrue(R.text_fallback('The provided passages do not specify the route. ["C1"]')['abstain'])

    def test_pick_label_by_product_rxcui(self):
        r = {'openfda': {'route': ['ORAL'], 'brand_name': ['Entresto'], 'generic_name': ['SACUBITRIL AND VALSARTAN'],
                         'rxcui': ['1656340']}, 'indications_and_usage': ['x'], 'effective_time': '2024'}
        self.assertIs(R.pick_label(None, [r], ['1656340']), r)
        self.assertIsNone(R.pick_label(None, [r], ['999']))


class Corpus(unittest.TestCase):
    def test_build_questions_and_chunks(self):
        corpus, qs = R.build(LABELS, COV)
        self.assertEqual(len(qs), 3 * 2 + 2)
        self.assertTrue(all(c['id'].startswith('C') for c in corpus))
        # Gold route metadata must not appear verbatim as a field in the corpus text.
        self.assertFalse(any('OPHTHALMIC' in c['text'] for c in corpus))
        self.assertEqual(sum(c['section'] == 'boxed_warning' for c in corpus), 2)

    def test_select_generics_cleans_cms_names(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 's.csv'
            p.write_text('generic_name,drug_cost\n"Insulin Glargine,Hum.Rec.Anlog",5\nApixaban,9\napixaban,1\n')
            self.assertEqual(R.select_generics(p, 5), ['apixaban', 'insulin glargine'])

    def test_retrieval_finds_right_drug(self):
        corpus, qs = R.build(LABELS, COV)
        names = {k: l['generic'] for l in LABELS for k in (l['generic'], l['brand'].lower())}
        ret = R.Retriever(corpus, hash_embed, names)
        for q in qs:
            for m in ('bm25', 'dense', 'hybrid', 'hybrid_drug_filter'):
                self.assertIn(q['doc'], [c['doc'] for c in ret.top(q['question'], m)], (m, q['question']))


class DrugFilter(unittest.TestCase):
    def test_filter_blocks_other_drugs_boxed_passages(self):
        corpus, qs = R.build(LABELS, COV)
        names = {k: l['generic'] for l in LABELS for k in (l['generic'], l['brand'].lower())}
        ret = R.Retriever(corpus, hash_embed, names)
        q = 'Does the Xalatan prescribing information carry a boxed warning?'
        self.assertEqual(ret.drugs_in(q), ({'latanoprost'}, 'exact'))
        self.assertTrue(any(c['section'] == 'boxed_warning' for c in ret.top(q, 'hybrid')))
        got = ret.top(q, 'hybrid_drug_filter')
        self.assertTrue(got and all(c['drug'] == 'latanoprost' for c in got))

    def test_misspelling_and_variants(self):
        corpus, qs = R.build(LABELS, COV)
        names = {k: l['generic'] for l in LABELS for k in (l['generic'], l['brand'].lower())}
        ret = R.Retriever(corpus, hash_embed, names)
        self.assertEqual(R.misspell('Eliquis'), 'Elqiuis')
        self.assertEqual(ret.drugs_in('Does the Elqiuis label carry a boxed warning?'), ({'apixaban'}, 'fuzzy'))
        got = ret.top('Does the Xaltaan prescribing information carry a boxed warning?', 'hybrid_drug_filter')
        self.assertTrue(all(c['drug'] == 'latanoprost' for c in got))
        v = R.variants(LABELS, qs)
        self.assertEqual({x['variant'] for x in v}, {'misspelled_brand', 'generic_name'})
        self.assertTrue(all(x['gold'] == next(q['gold'] for q in qs if q['type'] == x['type'] and q['drug'] == x['drug']) for x in v))

    def test_inferred_drug_without_name(self):
        corpus, _ = R.build(LABELS, COV)
        ret = R.Retriever(corpus, hash_embed, {})
        drugs, how = ret.resolve('one drop in the affected eye in the evening')
        self.assertEqual((drugs, how), ({'latanoprost'}, 'inferred'))

    def test_no_name_never_errors(self):
        corpus, _ = R.build(LABELS, COV)
        ret = R.Retriever(corpus, hash_embed, {'eliquis': 'apixaban'})
        q = 'which drug is taken once weekly?'
        drugs, how = ret.resolve(q)
        got = ret.top(q, 'hybrid_drug_filter')
        self.assertIn(how, ('inferred', 'unfiltered'))
        self.assertTrue(1 <= len(got) <= R.TOP_K)
        if how == 'unfiltered':
            self.assertEqual(len(got), R.TOP_K)
        else:
            self.assertTrue(all(c['drug'] in drugs for c in got))


class Batching(unittest.TestCase):
    def test_batch_prefetch_matches_sequential(self):
        calls = []

        class FakeBatch:
            def __init__(self): self.cache = {}
            def _call(self, msgs):
                calls.append(1)
                return {'answer': 'No.', 'citations': [], 'abstain': False}, 'raw'
            def batch(self, all_msgs):
                for m in all_msgs:
                    k = json.dumps(m, sort_keys=True)
                    if k not in self.cache:
                        self.cache[k] = self._call(m)
            def __call__(self, msgs):
                k = json.dumps(msgs, sort_keys=True)
                if k not in self.cache:
                    raise AssertionError('prompt was not prefetched')
                return self.cache[k]
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 's.csv'
            p.write_text('generic_name,drug_cost\nApixaban,9\nSemaglutide,8\nLatanoprost,7\n')
            R.run(t, p, labels=LABELS, cov=COV, qa=FakeBatch(), embed=hash_embed, boot=20)
        self.assertGreater(len(calls), 0)


class EndToEnd(unittest.TestCase):
    def test_run_with_injected_components(self):
        def qa(msgs):
            user = msgs[-1]['content']
            if 'Context:' not in user:
                return {'answer': 'I am not sure.', 'citations': [], 'abstain': True}, 'raw'
            first = user.split('[', 2)[1].split(']')[0]
            q = user.split('Question: ')[1]
            ctx = user.split('Question: ')[0]
            if 'route' in q:
                a = 'Orally.' if 'Eliquis' in ctx[:200] else 'Subcutaneously.' if 'Ozempic' in ctx[:200] else 'Into the eye.'
            elif 'boxed' in q:
                a = 'Yes.' if 'Boxed warning:' in ctx else 'No.'
            else:
                a = '87.3%' if 'semaglutide' in q else '0%'
            return {'answer': a, 'citations': [first, 'C9999'], 'abstain': False}, 'raw'
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 's.csv'
            p.write_text('generic_name,drug_cost\nApixaban,9\nSemaglutide,8\nLatanoprost,7\n')
            r = R.run(t, p, labels=LABELS, cov=COV, qa=qa, embed=hash_embed, boot=50)
            self.assertEqual(r['answer_accuracy']['rag_hybrid_drug_filter']['ALL']['accuracy'], 1.0)
            self.assertEqual(r['boxed_warning_ablation']['questions'], 3)
            self.assertGreater(r['robustness']['questions'], 0)
            self.assertEqual(r['answer_accuracy']['closed_book']['ALL']['abstained'], 8)
            self.assertEqual(r['citations']['answers_citing_unknown_ids'], 8)
            self.assertEqual(r['robustness']['misspelled_boxed_answers']['filtered']['false_yes_on_no_boxed_drugs'], 0)
            rows = [json.loads(l) for l in (Path(t) / 'drug_rag_predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 8)


if __name__ == '__main__':
    unittest.main()
