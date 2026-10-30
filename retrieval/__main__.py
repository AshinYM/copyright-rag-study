"""Run with python -m retrieval. Core search/evaluation uses standard library only."""
import argparse
import json
from pathlib import Path

from .data import import_workbooks, load_cases, save_new
from .evaluate import evaluate, review_queue
from .search import Search, METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    importing = commands.add_parser('import-xlsx')
    importing.add_argument('--corpus-xlsx', type=Path, required=True)
    importing.add_argument('--questions-xlsx', type=Path, required=True)
    importing.add_argument('--output', type=Path, required=True)
    for name in ['search', 'evaluate', 'review']:
        sub = commands.add_parser(name)
        sub.add_argument('--corpus', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
        if name == 'search':
            sub.add_argument('--question', required=True)
            sub.add_argument('--method', choices=METHODS, default='rrf_case_unique')
            sub.add_argument('--k', type=int, default=3)
        else:
            sub.add_argument('--questions', type=Path, required=True)
            if name == 'evaluate':
                sub.add_argument('--split', choices=['dev', 'test', 'review'], default='test')
                sub.add_argument('--k', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; existing results are never overwritten')
    if args.command == 'import-xlsx':
        print(json.dumps(import_workbooks(args.corpus_xlsx, args.questions_xlsx, args.output)))
        return
    search = Search(load_cases(args.corpus))
    if args.command == 'search':
        save_new(args.output, search.evidence(args.question, args.method, args.k))
    else:
        dataset = json.loads(args.questions.read_text())
        if args.command == 'review':
            save_new(args.output, review_queue(search, dataset))
        else:
            report, detail = evaluate(search, dataset, args.split, args.k)
            # A fresh output directory keeps row-level retrieval separate from public summary.
            args.output.mkdir(parents=True)
            save_new(args.output/'summary.json', report)
            save_new(args.output/'details.json', detail)
            save_new(args.output/'provenance.json', {'corpus_sha256': search.digest})
    print(json.dumps({'command': args.command, 'cases': len(search.cases), 'chunks': len(search.chunks), 'network_calls': 0}))


if __name__ == '__main__':
    main()
