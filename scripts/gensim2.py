import os 
from gensim import corpora, models, similarities
import codecs
from nltk.corpus import stopwords
import string

path_in = '/u/metanet/clustering/m4source/data/' #'c:/out_es_2/'
path_out = '/u/metanet/clustering/m4source/data/'
lang = 'spanish'
#class MyCorpus(object):
#    def __iter__(self):
#        for fname in return_flat_filenames(path_in):
#            yield dictionary.doc2bow(file2str(fname))
#replist = ['!', ',','.',':','1','2','3','4','5','6','7','8','9','0','-', '"', "'"]
#def file2str(fname):
#    with codecs.open(path_in + fname, 'r', 'utf-8') as file1:
#        str1 = ''
#        for line in file1:
#            str1 += line + ' '
#        str1 = replacer(str1, replist)
#    return str1.lower().split()

#def replacer(str1, list1):
#    for item in list1:
#        str1 = string.replace(str1, item, '')
#    return str1


#def return_flat_filenames(path):
#    for (dirpath, dirnames, filenames) in os.walk(path):
#        filenames = filenames
#    return filenames


#fnames = return_flat_filenames(path_in)
#print file2str(fnames[0])[1:2]
#corpus = MyCorpus()
#dictionary = corpora.Dictionary(file2str(fname) for fname in fnames)
#once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq < 80]
#stop2 = ['1', '2', '3', '4', '5', '6','7','8','9','10','11','12','13', '14', '15','16','17','18','19','20','21','22','23','24','25','26']
##stop_ids = [dictionary.token2id[stopword] for stopword in stop2]
#stop_ids = None #[dictionary.token2id[stopword] for stopword in stopwords.words(lang)]
#dictionary.filter_tokens(once_ids)#/ + stop_ids)
#dictionary.compactify()
#dictionary.save(path_out + 'dictionary_'+lang+'.dict')
dictionary = corpora.Dictionary.load(path_out + 'dictionary_'+lang+'2.dict')

corpus = corpora.MmCorpus(path_out+'corpus_'+lang+'2.mm')
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2e2)
corpus_lsi = lsi[corpus_tfidf]
lsi.save(path_out + 'model'+lang+'2.lsi')
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2e2)
corpus_lsi = lsi[corpus_tfidf]
lsi.save(path_out + 'model'+lang+'2.lsi')

def similarity(lsi, subdim, word):
    sims = [similarities.MatrixSimilarity(lsi[[dictionary.doc2bow([sword])]])[lsi[dictionary.doc2bow([word])]][0] for sword in subdim]
