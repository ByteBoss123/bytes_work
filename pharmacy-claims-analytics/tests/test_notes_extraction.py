"""Logic tests for the notes-extraction pipeline on tiny hand-written inputs (test inputs only)."""
import json
import tempfile
import unittest
from pathlib import Path

from pbm import notes_extraction as N


def row(text, spans):
    return {'text': text, 'spans': spans}


class Loading(unittest.TestCase):
    def test_masked_ids_restored_and_spans_trimmed(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'a.jsonl'
            p.write_text('{"id": *****, "text": "Timolol BID, OU", "meta": {}, "labels": [[0, 8, "DRUG"], [8, 11, "FREQUENCY"]]}\n'
                         'not json\n')
            rows, bad = N.load(p)
            self.assertEqual(bad, 1)
            self.assertEqual(rows[0]['spans'], [(0, 7, 'DRUG'), (8, 11, 'FREQUENCY')])

    def test_split_cuts_at_note_start(self):
        rows = [row('HPI: a', [])] + [row('x', [])] * 7 + [row('HPI: b', [])] + [row('y', [])] * 3
        train, test, blocks, info = N.split(rows, test_fraction=.4)
        self.assertEqual(info['cut_index'], 8)
        self.assertTrue(test[0]['text'].startswith('HPI:'))
        self.assertEqual(set(blocks), {1})


class Extraction(unittest.TestCase):
    def test_lexicon_learns_from_train_and_filters_ambiguous(self):
        train = [row('Timolol BID OU', [(0, 7, 'DRUG'), (8, 11, 'FREQUENCY'), (12, 14, 'ROUTE')]),
                 row('IOP 18 OU, CD 0.4 OU', [])]
        lex = N.Lexicon(train)
        self.assertIn('timolol', lex.entries['DRUG'])
        self.assertNotIn('ou', lex.entries['ROUTE'])  # labeled 1 of 3 times
        self.assertEqual(lex.predict('start timolol qhs'), [(6, 13, 'DRUG')])

    def test_llm_strings_mapped_and_hallucinations_counted(self):
        text = 'Latanoprost qhs OU and timolol BID OU'
        parsed = {'medications': [{'drug': 'latanoprost', 'frequency': 'qhs', 'route': 'OU'},
                                  {'drug': 'Timolol', 'frequency': 'bid', 'route': 'OU'},
                                  {'drug': 'Rhopressa', 'frequency': None, 'route': None}]}
        spans, missing = N.to_spans(text, parsed)
        self.assertIn((16, 18, 'ROUTE'), spans)
        self.assertIn((35, 37, 'ROUTE'), spans)  # second OU, not the first one again
        self.assertEqual(missing, [{'type': 'DRUG', 'text': 'Rhopressa'}])

    def test_score_strict_and_lenient(self):
        g = [row('Cosopt BID OU', [(0, 6, 'DRUG'), (7, 10, 'FREQUENCY')])]
        p = [[(0, 6, 'DRUG'), (7, 9, 'FREQUENCY'), (11, 13, 'ROUTE')]]
        s = N.score(g, p)
        self.assertEqual(s['DRUG']['f1'], 1.0)
        self.assertEqual(s['FREQUENCY']['recall'], 0.0)
        self.assertEqual(s['FREQUENCY']['lenient_recall'], 1.0)
        self.assertEqual(s['ROUTE']['precision'], 0.0)

    def test_gold_json_round_trip(self):
        r = row('Brimonidine BID, OU', [(0, 11, 'DRUG'), (12, 15, 'FREQUENCY'), (17, 19, 'ROUTE')])
        spans, missing = N.to_spans(r['text'], N.gold_json(r))
        self.assertEqual(sorted(spans), r['spans'])
        self.assertEqual(missing, [])
        json.dumps(N.messages(r['text'], [r]))


if __name__ == '__main__':
    unittest.main()
