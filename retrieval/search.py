"""Deterministic BM25 and character TF-IDF retrieval, RRF and MMR ablations."""
from collections import Counter, defaultdict
import math
import re
import unicodedata

from .data import case_key, case_numbers, chunk_cases, corpus_digest

METHODS = ('bm25_body', 'bm25_metadata', 'char_tfidf', 'rrf', 'rrf_case_unique', 'rrf_mmr_case_unique')
CONFIG = {'chunk_size': 1000, 'chunk_overlap': 150, 'candidate_window': 50,
          'bm25_k1': 1.5, 'bm25_b': .75, 'rrf_constant': 60, 'mmr_lambda': .7,
          'char_ngram': 2, 'semantic_embeddings': False}


def normalized(text):
    return unicodedata.normalize('NFKC', text).lower()


def words(text):
    # No morphological analyzer: Korean particle variants may not match.
    text = normalized(text)
    return re.findall(r'[가-힣a-z0-9]+', text)+sorted(case_numbers(text))


def grams(text):
    # Bigrams within words; no whitespace-spanning grams or pretrained model.
    output = []
    for token in re.findall(r'[가-힣a-z0-9]+', normalized(text)):
        output.extend(token[i:i+2] for i in range(len(token)-1))
    return output


def dot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(v*b.get(k, 0.) for k, v in a.items())


def unit(vector):
    norm = math.sqrt(sum(v*v for v in vector.values()))
    return {k: v/norm for k, v in vector.items()} if norm else {}


def fuse(lists, constant=60):
    if constant < 0:
        raise ValueError('RRF constant must be nonnegative')
    scores = defaultdict(float)
    for ranking in lists:
        seen = set()
        for rank, (key, _) in enumerate(ranking, 1):
            if key not in seen:
                scores[key] += 1/(constant+rank)
                seen.add(key)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class BM25:
    def __init__(self, texts):
        self.count = len(texts)
        self.lengths = []
        self.postings = defaultdict(list)
        for i, text in enumerate(texts):
            counts = Counter(words(text))
            self.lengths.append(sum(counts.values()))
            for token, count in counts.items():
                self.postings[token].append((i, count))
        self.average = sum(self.lengths)/self.count or 1.

    def scores(self, query):
        scores = defaultdict(float)
        for token in set(words(query)):
            postings = self.postings.get(token, [])
            idf = math.log(1+(self.count-len(postings)+.5)/(len(postings)+.5))
            for i, tf in postings:
                denominator = tf+1.5*(1-.75+.75*self.lengths[i]/self.average)
                scores[i] += idf*tf*2.5/denominator
        return scores


class Search:
    def __init__(self, cases, *, size=1000, overlap=150):
        self.records = {c.record_id: c for c in cases}
        self.cases = defaultdict(list)
        for case in cases:
            self.cases[case.case_id].append(case)
        if len(self.records) != len(cases) or not cases:
            raise ValueError('Non-empty unique records required; use validate_cases first')
        self.digest = corpus_digest(cases)
        self.chunks = chunk_cases(cases, size, overlap)
        self.config = {**CONFIG, 'chunk_size': size, 'chunk_overlap': overlap}
        self.body_bm25 = BM25([c.text for c in self.chunks])
        self.metadata_bm25 = BM25([c.header+'\n'+c.text for c in self.chunks])
        counts = [Counter(grams(c.header+'\n'+c.text)) for c in self.chunks]
        df = Counter(token for row in counts for token in row)
        self.idf = {token: math.log((1+len(counts))/(1+freq))+1 for token, freq in df.items()}
        self.vectors = [unit({k: (1+math.log(v))*self.idf[k] for k, v in row.items()}) for row in counts]
        self.char_postings = defaultdict(list)
        for i, vector in enumerate(self.vectors):
            for key, value in vector.items():
                self.char_postings[key].append((i, value))
        self.aliases = defaultdict(set)
        for case in cases:
            for alias in case_numbers(case.case_id) | {case.case_id}:
                self.aliases[alias].add(case.case_id)

    def char_scores(self, query):
        q = unit({k: (1+math.log(v))*self.idf[k] for k, v in Counter(grams(query)).items() if k in self.idf})
        scores = defaultdict(float)
        for token, value in q.items():
            for i, weight in self.char_postings[token]:
                scores[i] += value*weight
        return scores

    def retrieve(self, query, method='rrf_case_unique', k=3):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError('Question must contain 1..2000 characters')
        if method not in METHODS or type(k) is not int or not 1 <= k <= 20:
            raise ValueError('Unknown method or k outside 1..20')
        # Exact case-number routing is deliberately NOT used in the ablation.
        # Metadata BM25 provides case-number tokens; char retrieval may still
        # find partial matches, which must not be interpreted as an exact ID hit.
        def rank(scores):
            return sorted(((i, v) for i, v in scores.items() if v > 0), key=lambda p: (-p[1], self.chunks[p[0]].chunk_id))
        if method == 'bm25_body':
            ranking = rank(self.body_bm25.scores(query))
        elif method == 'bm25_metadata':
            ranking = rank(self.metadata_bm25.scores(query))
        elif method == 'char_tfidf':
            ranking = rank(self.char_scores(query))
        else:
            bm = rank(self.metadata_bm25.scores(query))[:50]
            ch = rank(self.char_scores(query))[:50]
            ranking = fuse([bm, ch])
        if method == 'rrf_mmr_case_unique':
            # MMR uses normalized RRF relevance and char TF-IDF similarity.
            # Both are ranking signals, never calibrated answer confidence.
            pool, selected, seen = list(ranking[:50]), [], set()
            maximum = max((v for _, v in pool), default=1.)
            while pool and len(selected) < k:
                candidates = [(i, v) for i, v in pool if self.chunks[i].case_id not in seen]
                if not candidates:
                    break
                def objective(pair):
                    i, v = pair
                    redundancy = max((dot(self.vectors[i], self.vectors[j]) for j, _ in selected), default=0.)
                    return -.7*v/maximum+.3*redundancy, self.chunks[i].chunk_id
                chosen = min(candidates, key=objective)
                selected.append(chosen)
                seen.add(self.chunks[chosen[0]].case_id)
                pool.remove(chosen)
            ranking = selected
        elif method == 'rrf_case_unique':
            selected, seen = [], set()
            for i, value in ranking:
                if self.chunks[i].case_id not in seen:
                    selected.append((i, value))
                    seen.add(self.chunks[i].case_id)
            ranking = selected
        return [(self.chunks[i], float(value)) for i, value in ranking[:k]]

    def resolve_case(self, reference):
        matches = self.aliases.get(case_key(reference), set())
        if len(matches) != 1:
            raise ValueError('Relevant case ID is missing or ambiguous in corpus')
        return next(iter(matches))

    def evidence(self, query, method='rrf_case_unique', k=3):
        hits = self.retrieve(query, method, k)
        sources = []
        for chunk, score in hits:
            original = self.records[chunk.record_id].text
            if original[chunk.start:chunk.end] != chunk.text:
                raise ValueError('Source offset integrity failure')
            sources.append({'chunk_id': chunk.chunk_id, 'case_id': chunk.case_id, 'record_id': chunk.record_id,
                            'title': chunk.title, 'start': chunk.start, 'end': chunk.end,
                            'text': chunk.text, 'score': score})
        return {'status': 'matches_found' if sources else 'no_lexical_match',
                'generated_answer': False, 'legal_sufficiency_verified': False,
                'score_is_confidence': False, 'method': method,
                'message': '원문 검색 결과입니다. 질문에 대한 충분한 근거인지는 원문 검토가 필요합니다.' if sources else '검색어와 겹치는 근거를 찾지 못했습니다.',
                'sources': sources}
