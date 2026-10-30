"""Case-level relevance metrics, incomplete-judgment accounting and review queue."""
import math
from statistics import mean

from .data import case_key
from .search import METHODS, normalized


def validate_questions(dataset, search):
    if not isinstance(dataset, dict) or dataset.get('kind') not in {'synthetic', 'reviewed', 'legacy_sanity', 'unjudged'}:
        raise ValueError('Question dataset requires kind: synthetic/reviewed/legacy_sanity/unjudged')
    items = dataset.get('items')
    if not isinstance(items, list) or not items:
        raise ValueError('Non-empty question items required')
    output, ids, questions = [], set(), set()
    for row in items:
        if not isinstance(row, dict):
            raise ValueError('Question must be an object')
        for field in ['id', 'question', 'split', 'judgment']:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError('Question requires nonempty id/question/split/judgment')
        if row['id'] in ids or row['split'] not in {'dev', 'test', 'review'} or len(row['question']) > 2000:
            raise ValueError('Duplicate question ID, invalid split or oversized question')
        ids.add(row['id'])
        query_key = ' '.join(normalized(row['question']).split())
        if query_key in questions:
            raise ValueError('Repeated normalized question; avoid duplicate rows across evaluation splits')
        questions.add(query_key)
        judgment = row['judgment']
        if judgment not in {'relevant', 'unanswerable', 'unjudged'}:
            raise ValueError('Invalid judgment')
        references = row.get('relevant_cases', [])
        if not isinstance(references, list) or any(not isinstance(x, str) or not x.strip() for x in references):
            raise ValueError('relevant_cases must be a string list')
        resolved = [search.resolve_case(x) for x in references]
        if len(resolved) != len(set(resolved)):
            raise ValueError('Duplicate relevant case or aliases')
        evidence = row.get('evidence', [])
        if not isinstance(evidence, list):
            raise ValueError('evidence must be a list')
        if judgment == 'relevant' and not resolved:
            raise ValueError('Relevant question needs at least one reference')
        if judgment != 'relevant' and (resolved or evidence):
            raise ValueError('Unjudged/unanswerable questions must not carry relevance labels')
        if dataset['kind'] == 'unjudged' and judgment != 'unjudged':
            raise ValueError('Unjudged dataset cannot contain judged rows')
        if judgment != 'unjudged' and dataset['kind'] != 'legacy_sanity':
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ['reviewer', 'rationale']):
                raise ValueError('Judgment requires reviewer and rationale provenance')
        checked = []
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {'case_id', 'quote'} or not all(isinstance(v, str) and v.strip() for v in item.values()):
                raise ValueError('Evidence requires case_id and nonempty exact quote')
            case_id = search.resolve_case(item['case_id'])
            if case_id not in resolved or not any(item['quote'] in c.text for c in search.cases[case_id]):
                raise ValueError('Evidence must be an exact source span of a relevant case')
            checked.append({'case_id': case_id, 'quote': item['quote']})
        if judgment == 'relevant' and dataset['kind'] != 'legacy_sanity' and set(resolved) != {x['case_id'] for x in checked}:
            raise ValueError('Each relevant case needs at least one source evidence quote')
        output.append({**row, 'relevant_cases': resolved, 'evidence': checked})
    return output


def rank_metrics(retrieved, relevant, k):
    """Duplicates consume rank slots, but earn relevance gain at most once."""
    if type(k) is not int or k <= 0:
        raise ValueError('k must be a positive integer')
    relevant = set(relevant)
    if not relevant:
        raise ValueError('Do not score unknown or unanswerable labels as a negative gold set')
    selected, gains, ranks = retrieved[:k], [], []
    seen = set()
    for rank, case_id in enumerate(selected, 1):
        hit = case_id in relevant and case_id not in seen
        gains.append(int(hit))
        if hit:
            ranks.append(rank)
        seen.add(case_id)
    dcg = sum(g/math.log2(i+2) for i, g in enumerate(gains))
    ideal = sum(1/math.log2(i+2) for i in range(min(len(relevant), k)))
    return {'hit_at_k': float(bool(ranks)), 'recall_at_k': sum(gains)/len(relevant),
            'precision_at_k': sum(gains)/k,
            'mrr_at_k': 1/ranks[0] if ranks else 0., 'ndcg_at_k': dcg/ideal}


def evaluate(search, dataset, split='test', k=3):
    items = [q for q in validate_questions(dataset, search) if q['split'] == split]
    if not items:
        raise ValueError('No questions in requested split')
    detail, results = [], []
    for method in METHODS:
        rows = []
        for q in items:
            hits = search.retrieve(q['question'], method, k)
            case_ids = [c.case_id for c, _ in hits]
            duplicate_rate = 1-len(set(case_ids))/len(case_ids) if case_ids else 0.
            values = rank_metrics(case_ids, q['relevant_cases'], k) if q['judgment'] == 'relevant' else None
            span = None
            if q['evidence']:
                span = sum(any(c.case_id == e['case_id'] and e['quote'] in c.text for c, _ in hits) for e in q['evidence'])/len(q['evidence'])
            row = {'method': method, 'question_id': q['id'], 'judgment': q['judgment'],
                   'retrieved_count': len(hits), 'unique_case_count': len(set(case_ids)),
                   'duplicate_rate': duplicate_rate, 'metrics': values,
                   'evidence_quote_coverage_at_k': span,
                   'returned_on_unanswerable': bool(hits) if q['judgment'] == 'unanswerable' else None,
                   'retrieved_case_ids': case_ids}
            rows.append(row)
        relevant = [r for r in rows if r['metrics'] is not None]
        negatives = [r for r in rows if r['judgment'] == 'unanswerable']
        spans = [r['evidence_quote_coverage_at_k'] for r in rows if r['evidence_quote_coverage_at_k'] is not None]
        summary = {'method': method, 'total_questions': len(rows), 'relevance_scored_questions': len(relevant),
                   'unanswerable_questions': len(negatives),
                   'unjudged_questions_excluded_from_quality_scores': sum(r['judgment'] == 'unjudged' for r in rows),
                   'relevance_metrics': {key: mean(r['metrics'][key] for r in relevant) for key in ['hit_at_k', 'recall_at_k', 'precision_at_k', 'mrr_at_k', 'ndcg_at_k']} if relevant else None,
                   'avg_unique_cases': mean(r['unique_case_count'] for r in rows),
                   'avg_duplicate_rate': mean(r['duplicate_rate'] for r in rows),
                   'empty_result_rate': mean(r['retrieved_count'] == 0 for r in rows),
                   'unanswerable_with_any_result_rate': mean(r['returned_on_unanswerable'] for r in negatives) if negatives else None,
                   'evidence_quote_coverage_at_k': mean(spans) if spans else None}
        results.append(summary)
        detail.extend(rows)
    # Pair comparisons are diagnostics only, never automatic retriever selection.
    baseline = {r['question_id']: r for r in detail if r['method'] == 'bm25_body'}
    changes = []
    for row in detail:
        if row['method'] == 'bm25_body' or row['metrics'] is None:
            continue
        before = baseline[row['question_id']]['metrics']
        delta = row['metrics']['mrr_at_k']-before['mrr_at_k']
        if delta:
            changes.append({'method': row['method'], 'question_id': row['question_id'], 'mrr_delta': delta})
    summary = {'dataset_kind': dataset['kind'], 'split': split, 'k': k,
               'corpus_cases': len(search.cases), 'corpus_records': len(search.records), 'chunks': len(search.chunks), 'config': search.config,
               'quality_scope': 'synthetic_fixture_only' if dataset['kind'] == 'synthetic' else 'legacy_development_sanity_not_independent_gold' if dataset['kind'] == 'legacy_sanity' else 'user_supplied_judgments' if dataset['kind'] == 'reviewed' else 'operational_only_no_relevance_labels',
               'semantic_or_answer_quality_measured': False,
               'default_method_preselected': 'rrf_case_unique',
               'automatic_winner_selection': False,
               'methods': results, 'mrr_changes_vs_body_bm25': changes}
    return summary, detail


def review_queue(search, dataset):
    rows = validate_questions(dataset, search)
    return {'corpus_sha256': search.digest, 'note': 'Candidates are retrieval output, not relevance labels. Review the full source independently.',
            'items': [{**row, 'candidates': {method: search.evidence(row['question'], method)['sources'] for method in METHODS}}
                      for row in rows]}
