from collections import Counter
import pickle
from pyserini.search import SimpleSearcher

logtxt = open('./prints.txt','w')

def retrievability_experiment(searcher, c, queryFilePaths_and_noOfQueries_tuple):
    r_d = Counter()     # to store lucene_docids and their cumulative counts
    
    def retrieve(query, c, r_d):
        # getting the top c search results using the searcher
        hits = searcher.search(query,c)

        lucene_docids = []
        for hit in hits:
            lucene_docids.append(hit.lucene_docid)

        r_d.update(lucene_docids)
    
    for item in queryFilePaths_and_noOfQueries_tuple:
        filePath = item[0]
        len_queries = item[1]
        if 'unigram' in filePath:
            query_type = 'Unigram queries'
        elif 'bigram' in filePath:
            query_type = 'Bigram queries'
        
        # print(f"{query_type} run starts...")
        logtxt.write(f"{query_type} run starts...")
        with open(filePath) as f:
            i = 0
            for line in f:
                query = line[:-1]
                retrieve(query, c, r_d)
                
                i += 1
                if i%(len_queries//20)==0:
                    # print(f"{query_type} progress... {i*100/len_queries: .2f}%")
                    logtxt.write(f"{query_type} progress... {i*100/len_queries: .2f}%")
            # print(f'Run completed with {query_type} progress... {i*100/len_queries: .2f}%')
            logtxt.write(f'Run completed with {query_type} progress... {i*100/len_queries: .2f}%')

    with open(f'./rd_dumps/rd_queryset2_{model}_{c}.pickle', 'wb') as f:
        pickle.dump(r_d, f)
        
    return r_d

searcher = SimpleSearcher.from_prebuilt_index('robust04')

model = 'bm25+rm3'
k1 = 0.7; b = 0.35
searcher.set_bm25(k1, b)
docs = 15; N = 10; origweight = 0.4
searcher.set_rm3(docs, N, origweight)

query_set_files = ['./Queries/unigram_queries_isalpha_tf5_not-analyzed_pos-filtered.txt', './Queries/bigram_queries_isalpha_tf20_not-analyzed_pos-filtered.txt']
queryPath_len_tuple = [(path, sum(1 for _ in open(path))) for path in query_set_files]

all_rd = dict()
c_list  = [10,20,30,50,100]
for c in c_list:
    # print(f'{model} retrievals starting for c = {c}\n')
    logtxt.write(f'{model} retrievals starting for c = {c}\n')
    all_rd[f'r_d_{model}_{c}'] = retrievability_experiment(searcher,c,queryPath_len_tuple)
    # print(f'Completed! c = {c}\n')
    logtxt.write(f'Completed! c = {c}\n')
    
logtxt.close()