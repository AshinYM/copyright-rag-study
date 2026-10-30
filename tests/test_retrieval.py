from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from retrieval.data import (case_key, chunk_cases, import_workbooks, load_cases,
                            save_new, validate_cases)
from retrieval.evaluate import evaluate, rank_metrics, validate_questions
from retrieval.search import METHODS, Search, fuse

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_input_schema_and_record_identity(self):
        for records in [[], [{'case_id': 'x', 'title': 'x', 'text': ''}],
                        [{'case_id': 'x', 'title': 'x', 'text': 'x', 'extra': 1}]]:
            with self.assertRaises(ValueError):
                validate_cases(records)
        self.assertEqual(case_key(' ２０１７ 다 ２１２０９５ '), '2017다212095')
        row = {'case_id': 'A', 'title': 'a', 'text': 'alpha'}
        with self.assertRaises(ValueError):
            validate_cases([row, row])
        result = validate_cases([{**row, 'record_id': 'source-1'}, {**row, 'record_id': 'source-2', 'text': 'beta'}])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].case_id, result[1].case_id)

    def test_chunks_preserve_offsets_overlap_and_coverage(self):
        cases = validate_cases([{'case_id': 'A', 'title': 'a', 'text': ' 가나다\n라마바사\n아자차카 타파하 '}])
        chunks = chunk_cases(cases, 8, 3)
        covered = set()
        for chunk in chunks:
            self.assertEqual(chunk.text, cases[0].text[chunk.start:chunk.end])
            self.assertLessEqual(len(chunk.text), 8)
            covered.update(range(chunk.start, chunk.end))
        self.assertEqual(covered, set(range(len(cases[0].text))))
        for size, overlap in [(0, 0), (8, 8), (8, -1)]:
            with self.assertRaises(ValueError):
                chunk_cases(cases, size, overlap)

    def test_import_preserves_variants_skips_blank_and_deduplicates_exact_rows(self):
        corpus = [('사건명', '사건번호', '선고일자', '법원명', '판례내용'),
                  ('title', 'A', 'date', 'court', 'body'), (None,)*5,
                  ('title', 'A', 'date', 'court', 'body'),
                  ('other', 'A', 'date', 'court', 'different body')]
        questions = [('ID', 'Category', 'Question'), ('E1', 'category', 'question')]
        closed = []
        class Book:
            def __init__(self, rows):
                self.worksheets = [types.SimpleNamespace(iter_rows=lambda **k: iter(rows))]
            def close(self): closed.append(True)
        fake = types.SimpleNamespace(load_workbook=lambda p, **kw: Book(corpus if p == 'corpus' else questions))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'openpyxl': fake}):
            path = Path(directory)/'import'
            audit = import_workbooks('corpus', 'questions', path)
            self.assertEqual(audit['preserved_records'], 2)
            self.assertEqual(audit['unique_case_ids'], 1)
            self.assertEqual(audit['ignored_fully_empty_rows'], 1)
            self.assertEqual(audit['removed_exact_full_row_duplicates'], 1)
            self.assertEqual(len(load_cases(path/'cases.json')), 2)
            self.assertEqual(json.loads((path/'questions-pending.json').read_text())['kind'], 'unjudged')
            with self.assertRaises(FileExistsError):
                import_workbooks('corpus', 'questions', path)
        self.assertEqual(len(closed), 2)

    def test_formula_and_partial_records_are_not_silently_skipped(self):
        for row in [('x', 'A', 'date', 'court', '=external()'), ('x', 'A', 'date', 'court', None)]:
            rows = [('사건명', '사건번호', '선고일자', '법원명', '판례내용'), row]
            fake = types.SimpleNamespace(load_workbook=lambda *a, **k: types.SimpleNamespace(
                worksheets=[types.SimpleNamespace(iter_rows=lambda **k: iter(rows))], close=lambda: None))
            with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'openpyxl': fake}):
                with self.assertRaises(ValueError):
                    import_workbooks('corpus', 'questions', Path(directory)/'out')

    def test_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'result.json'
            save_new(p, {'original': True})
            with self.assertRaises(FileExistsError): save_new(p, {'replacement': True})
            self.assertEqual(json.loads(p.read_text()), {'original': True})


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search = Search(load_cases(ROOT/'examples/cases.json'))

    def test_rrf_uses_rank_and_handles_duplicates(self):
        first = fuse([[(1, 100.), (2, .1)], [(2, 500.), (1, 0.)]])
        second = fuse([[(1, 1.), (2, .9)], [(2, .2), (1, .1)]])
        self.assertEqual(first, second)
        self.assertEqual(fuse([[(1, 1.), (1, 1.)]]), [(1, 1/61)])

    def test_case_uniqueness_and_max_k(self):
        for method in ['rrf_case_unique', 'rrf_mmr_case_unique']:
            hits = self.search.retrieve('글꼴 프로그램 코드', method, 3)
            ids = [c.case_id for c, _ in hits]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertLessEqual(len(hits), 3)

    def test_no_match_is_empty_and_not_synthesized(self):
        for method in METHODS:
            result = self.search.evidence('zzzzzzzz qqqqqqqq', method)
            self.assertEqual(result['sources'], [])
            self.assertEqual(result['status'], 'no_lexical_match')
            self.assertFalse(result['generated_answer'])
        result = self.search.evidence('온라인 게임')
        self.assertTrue(result['sources'])
        self.assertFalse(result['legal_sufficiency_verified'])
        self.assertFalse(result['score_is_confidence'])

    def test_citation_spans_resolve_to_correct_source_record(self):
        result = self.search.evidence('글꼴 프로그램')
        for item in result['sources']:
            record = self.search.records[item['record_id']]
            self.assertEqual(item['text'], record.text[item['start']:item['end']])
        chunk = self.search.chunks[0]
        with patch.object(self.search, 'retrieve', return_value=[(replace(chunk, text='tampered'), 1.)]):
            with self.assertRaisesRegex(ValueError, 'integrity'):
                self.search.evidence('query')

    def test_invalid_queries_and_methods(self):
        for query in ['', ' ', 'x'*2001]:
            with self.assertRaises(ValueError): self.search.retrieve(query)
        for method, k in [('wrong', 3), ('rrf', 0), ('rrf', 21)]:
            with self.assertRaises(ValueError): self.search.retrieve('query', method, k)

    def test_deterministic_ranking_independent_of_input_row_order(self):
        records = json.loads((ROOT/'examples/cases.json').read_text())
        reversed_search = Search(validate_cases(records[::-1]))
        for method in METHODS:
            self.assertEqual(self.search.retrieve('프로그램 표현', method), reversed_search.retrieve('프로그램 표현', method))

    def test_no_network_needed_even_if_network_is_disabled(self):
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')):
            index = Search(load_cases(ROOT/'examples/cases.json'))
            self.assertTrue(index.evidence('사진 촬영')['sources'])


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search = Search(load_cases(ROOT/'examples/cases.json'))
        cls.dataset = json.loads((ROOT/'examples/questions.json').read_text())

    def test_multi_relevant_metrics_do_not_reward_duplicate_case_slots(self):
        values = rank_metrics(['a', 'a', 'b'], ['a', 'b'], 3)
        self.assertEqual(values['recall_at_k'], 1.)
        self.assertEqual(values['precision_at_k'], 2/3)
        self.assertEqual(values['mrr_at_k'], 1.)
        self.assertLess(values['ndcg_at_k'], 1.)
        self.assertEqual(rank_metrics(['a'], ['a', 'b'], 3)['recall_at_k'], .5)
        self.assertEqual(rank_metrics(['a'], ['a'], 3)['precision_at_k'], 1/3)
        with self.assertRaises(ValueError): rank_metrics([], [], 3)

    def test_unjudged_is_not_a_zero_or_a_quality_score(self):
        dataset = {'kind': 'unjudged', 'items': [{'id': 'U1', 'question': '사진', 'split': 'review', 'judgment': 'unjudged'}]}
        summary, _ = evaluate(self.search, dataset, 'review')
        for row in summary['methods']:
            self.assertIsNone(row['relevance_metrics'])
            self.assertIsNone(row['unanswerable_with_any_result_rate'])
            self.assertEqual(row['unjudged_questions_excluded_from_quality_scores'], 1)

    def test_split_exclusion_and_unanswerable_failure_are_explicit(self):
        summary, detail = evaluate(self.search, self.dataset, 'test')
        self.assertTrue(all(not r['question_id'].startswith('D') for r in detail))
        for row in summary['methods']:
            self.assertEqual(row['relevance_scored_questions'], 7)
            self.assertEqual(row['unanswerable_questions'], 2)
            self.assertEqual(row['unjudged_questions_excluded_from_quality_scores'], 1)
            self.assertGreater(row['unanswerable_with_any_result_rate'], 0.)
        self.assertFalse(summary['automatic_winner_selection'])

    def test_bad_labels_spans_and_duplicate_ids_rejected(self):
        row = dict(self.dataset['items'][0])
        variants = [{**row, 'relevant_cases': []},
                    {**row, 'relevant_cases': ['not-in-corpus']},
                    {**row, 'evidence': []},
                    {**row, 'evidence': [{'case_id': 'DEMO-FONT', 'quote': 'invented text'}]},
                    {**row, 'reviewer': ''}, {**row, 'judgment': 'unjudged'}]
        for variant in variants:
            with self.assertRaises(ValueError):
                validate_questions({'kind': 'reviewed', 'items': [variant]}, self.search)
        with self.assertRaises(ValueError):
            validate_questions({'kind': 'synthetic', 'items': [row, row]}, self.search)

    def test_duplicate_question_across_dev_test_rejected(self):
        row = dict(self.dataset['items'][0])
        with self.assertRaisesRegex(ValueError, 'Repeated normalized question'):
            validate_questions({'kind': 'synthetic', 'items': [row, {**row, 'id': 'copy', 'split': 'test', 'question': '  '+row['question']+'  '}]}, self.search)

    def test_joint_case_alias_ambiguity_is_not_guessed(self):
        index = Search(validate_cases([
            {'case_id': '2017다123,2017다456', 'title': 'joint', 'text': 'first'},
            {'case_id': '2017다123', 'title': 'single', 'text': 'second'},
        ]))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            index.resolve_case('2017다123')

    def test_cli_roundtrip_on_public_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'evaluation'
            command = [sys.executable, '-m', 'retrieval', 'evaluate', '--corpus', str(ROOT/'examples/cases.json'),
                       '--questions', str(ROOT/'examples/questions.json'), '--split', 'test', '--output', str(output)]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
            actual = json.loads((output/'summary.json').read_text())
            expected, _ = evaluate(self.search, self.dataset, 'test')
            self.assertEqual(actual, expected)
            again = subprocess.run(command, cwd=ROOT, capture_output=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(json.loads((output/'summary.json').read_text()), expected)


if __name__ == '__main__':
    unittest.main()
