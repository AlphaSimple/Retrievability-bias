from collections import Counter
import pickle
from pyserini.search import SimpleSearcher


def retrieve(query, k=100):
    # getting the top k search results using the searcher
    return searcher.search(query,k)


searcher = SimpleSearcher.from_prebuilt_index('robust04')

model = 'bm25+rm3'
k1 = 0.7; b = 0.35
searcher.set_bm25(k1, b)
docs = 15; N = 10; origweight = 0.4
searcher.set_rm3(docs, N, origweight)

from multiprocessing.pool import Pool
pool = Pool()

print('Just testing flush True for log', flush=True)

all_rd = dict()
c_list  = [20,30,50,100]
for c in c_list:
    all_rd[f'r_d_{model}_{c}'] = Counter()      # to store lucene_docids and their cumulative counts
    
with open('./Queries/unigram_queries_isalpha_tf5_not-analyzed_pos-filtered.pickle', 'rb') as f:
    unigram_queries = pickle.load(f)
    
with open('./Queries/bigram_queries_isalpha_tf20_not-analyzed_pos-filtered.pickle', 'rb') as f:
    bigram_queries = pickle.load(f)
    
queries = unigram_queries + bigram_queries
    
for hits in pool.map(retrieve, queries):
    lucene_docids = []
    rank = 0
    for hit in hits:
        rank += 1
        lucene_docids.append(hit.lucene_docid)
        for c in c_list[:-1]:   # except c = 100
            if rank == c:
                all_rd[f'r_d_{model}_{c}'].update(lucene_docids)
    all_rd[f'r_d_{model}_{100}'].update(lucene_docids)  # c = 100
    
with open(f'./rd_dumps/rd_queryset2_{model}_c_20_30_50_100.pickle', 'wb') as f:
    pickle.dump(all_rd, f)
