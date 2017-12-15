#
# @author: E.D. Gutierrez
#
# subdims_matcher: matches input LM source words to the correct
# IARPA source subdimension, given an IARPA source dimension
#
# Main Routine: run1
#   
#   Default usage: run1(input_LMs, input_dims)
# 
#   Required Inputs: 
#     -input_LMs: a file of LM source words separated by whitespace or newlines
#     -input_dims: a file of IARPA souce dimensions separated by whitespace or
#           newlines;the n-th source dimension is assigned to the n-th LM source
#           word
#
#   (Optional Inputs; don't worry about these; they're set by default)
#     -master_dim_list: a two-level Python dictionary, where each entry
#           master_dim_list[dimension][subdimension] is a list of strings
#           with containing seed words for each IARPA subdimension
#     -BNC_mat: a matrix where each row is the semantic vector for a word in
#           a corpus
#     -BNC_vocab: a Python list containing the word labels for each row of
#           BNC_mat
# 
#    Output:
#     -best_dims: a list of 3-tuples.  Each tuple contains:
#           * the input source LM in the first position
#           * the input IARPA source dimension in the second position
#           * the predicted IARPA source subdimension in the third position
#
#
from sys import argv
import numpy
import scipy.spatial.distance
import scipy.io
import random
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.stem.snowball
import csv

lang = 'english'# 'russian', 'spanish'


def lemmatize(word, lemm = True):
    if lemm:
        if lang == 'english':
            word = nltk.stem.snowball.EnglishStemmer(True).stem(WordNetLemmatizer().lemmatize(word))
        elif lang=='spanish':
            word = nltk.stem.snowball.SpanishStemmer(True).stem(word)
        else:
            word = nltk.stem.snowball.RussianStemmer(True).stem(word)
    else:
        0
    return word.lower()

def file_to_list(filename, lemm = True):
    file1 = open(filename, 'r')
    list1 = []
    for line in file1:
        word1 = ''
        for word in line.split():
            word1 += ' ' + lemmatize(word, lemm)
        list1.append(word1[1:])
    return list1
    
def file_to_dict(filename, lemm = True):
    dim_list = {}
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        count = 0
        for row in reader:
            try:
                dim_list[row[0].lower()][row[1].lower()] = []
            except:
                dim_list[row[0].lower()] = {}
                dim_list[row[0].lower()][row[1].lower()] = []
            words = row[2].split('||')
            for word in words:
                dim_list[row[0].lower()][row[1].lower()].append(lemmatize(word, lemm))
    return dim_list




## DEFAULT GLOBAL VARIABLES
def_dist = scipy.spatial.distance.cosine
in_dir = 'C:/BNC_data/' #/u/metanet/clustering/m4source/'
input_LMs_file = in_dir+'test_list.txt' 
input_dims_file = in_dir + 'test_list_dims.txt'
BNC_vocab = file_to_list(in_dir+'lexicon.txt', lemm = False)
BNC_mat = scipy.io.loadmat(in_dir +'termtermmatrix.mat')
BNC_mat = BNC_mat['A']
master_dim_list = file_to_dict(in_dir+'master_dim_list.csv')


def levstn(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        a,b = b,a
        n,m = m,n
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]

def avg_dist(vec, mat, dist=def_dist, mean = numpy.mean):
    def median_or_greater(list1):
        list2 = numpy.zeros(len(list1))
        counter = 0
        if len(list1)>2:
            for item in list1:
                if item>=numpy.median(list1):
                    list2[counter] = item
                    counter += 1
            return list2[:(counter-1)]
        else:
            return list1
    if dist==levstn:
        d1 = numpy.zeros(len(mat))
        for row in range(len(mat)):
            d1[row] = numpy.exp(-2*dist(mat[row], vec))
            if d1[row] > (1 - 1e-3):
                d1[row] = 100
    else:
        d1 = numpy.zeros(mat.shape[0])
        for row in range(mat.shape[0]):
            d1[row] = -dist(mat[row,:], vec) #numpy.exp(-2*numpy.power(dist(mat[row, :], vec),1))
    d2 = -mean(median_or_greater(median_or_greater(d1)))
    if numpy.isnan(d2):
        print str(d2.shape[0])
        stop
    return d2

def populate_matrix(words, BNC_mat, BNC_vocab, random_guess=False, verbose = False):
    out_mat = numpy.zeros((len(words), BNC_mat.shape[1]))
    count = 0
    for word in words:
        try:
            out_mat[count, :] = BNC_mat[BNC_vocab.index(word), :]
            count += 1
        except:
            if random_guess:
                out_mat[count, :] = BNC_mat[random.choice(range(200)),:]
                count += 1
            else:
                0
            if verbose:
                print "Subdim word not found in BNC: " + str(word) + '\n'
    return out_mat[:(count-1),:]
    
def create_dim_model(dims, BNC_mat, BNC_vocab):
    dim_mat = {}
    for dim in dims:
        dim_mat[dim] = {}
        for subdim in dims[dim]:
            dim_mat[dim][subdim] = populate_matrix(dims[dim][subdim], BNC_mat, BNC_vocab)    
    return dim_mat

def pick_subdim(word, dim, IARPA_dim_mat, BNC_mat, master_dim_list, BNC_vocab):
    try:
        vec = BNC_mat[BNC_vocab.index(word), :] # or include some combination with the CM info here?
        mindist = 1e10
        for subdim in master_dim_list[dim]:
            dist1 = avg_dist(vec, IARPA_dim_mat[dim][subdim])
            print word + ' ' + subdim +' ' + ("%.2f" % dist1) +'\n'
            if dist1<mindist:
                best_dims = (word, dim, subdim)
                mindist = dist1
    except:
        mindist = 1e10
        print "Word not found in BNC:" + word + '\n'
        for subdim in master_dim_list[dim]:
            dist1 = avg_dist(word, master_dim_list[dim][subdim], levstn, numpy.max)
            print word + ' ' + subdim +' ' + ("%.2f" % dist1) +'\n'  
            if dist1 < mindist:
                best_dims = (word, dim, subdim)
    return best_dims

def run1(lang = 'english', input_LMs_file = input_LMs_file, input_dims_file = input_dims_file, master_dim_list = master_dim_list, IARPA_dim_mat = None, BNC_mat = BNC_mat, BNC_vocab = BNC_vocab):
    input_LMs = file_to_list(input_LMs_file)
    input_dims = file_to_list(input_dims_file, lemm = 0)
    IARPA_dim_mat = create_dim_model(master_dim_list, BNC_mat, BNC_vocab)
    LMs_dim_mat = populate_matrix(input_LMs, BNC_mat, BNC_vocab, random_guess=0)
    best_dims = []
    for (word, dim) in zip(input_LMs, input_dims):
        word = word.split()[0]
        print dim
        dim1 = dim.split()
        dim2 = ''
        for word1 in dim1:
            dim2 += '_' + word1
        dim = dim2[1:]
        best_dims.append(pick_subdim(word, dim, IARPA_dim_mat, BNC_mat, master_dim_list, BNC_vocab))
    return best_dims

def print_eval(ans):
    success, counter = 0, 0
    truth = file_to_list(in_dir + 'true_subdims.txt', False)
    with open(in_dir + 'eval.txt', 'w') as outfile:
        for (item, trueitem) in zip(ans, truth):
            if item[2]==trueitem:
                success += 1
            else:
                outfile.write(item[0] + ', ' + item[1] +', ' + item[2]+', ' + trueitem + '\n')
            counter += 1
    return (success, counter)
b = run1()
a = print_eval(b)
