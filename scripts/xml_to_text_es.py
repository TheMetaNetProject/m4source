#takes a corpus in directory path and removes XML tags, stopwords, punctuation, and uppercases, and then lemmatizes and removes stems
# Returns the cleaned corpus in a path specified by out_path
# author: E.D. Gutierrez
# Modified: 16 September 2013
from lxml import etree
from os import walk, sep
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import string
import re
lang = 'spanish'
stemmer = SnowballStemmer(lang)
#lemmatizer = WordNetLemmatizer()

def xml_to_text():
    path = 'c:/spanish_corpus/'
    out_path = 'c:/spanish_corpus_processed/'
    count_poetic = 0
    count_non_poetic = 0
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            if True: #filename[-4:] == '.':
                print(filename+' ')
                filenames = split_text(dirpath+'/'+filename, out_path)
                poetic = False # remove_poems(dirpath+'/'+filename)
                for name in filenames:
                    filepath = etree.parse(name)
                    notags = etree.tostring(filepath, encoding='utf8', method='text')
                    notags_tokens = remove_stopwords(notags[notags.find('\n'):].lower().split())  # Remove the BNC header
                    notags = ' '.join(notags_tokens)
                    outfile = open(name+'.txt','w')
                    outfile.write(notags)
                    outfile.close()
                    count_non_poetic += 1
                    count_poetic +=1
    print 'oui: ' + str(count_poetic) + 'non: ' + str(count_non_poetic)
                    

def split_text(file_in, out_path):
    with open(file_in, 'r') as infile:
        counter = 0
        str1 = ''
        filenames = []
        for line in infile.readlines():
            str1 = str1 + line
            if line.find('<DOC id=')>-1:
                filename = line[line.find('<DOC id=')+9: line.find('" type')]
            re1 = (line.find('</DOC>')>-1)
            if re1:
                open(out_path + filename, 'w').write(str1)
                str1 = ''
                filenames.append(out_path + filename)
    return filenames

def remove_stopwords(list1, stem=1):
    for index, token in enumerate(list1):
        token = remove_punctuation(token)
        if (token not in stopwords.words(lang))&is_ascii(token):
            if stem==1:
                list1[index] = stemmer.stem(token)
            else:
                list1[index] = lemmatizer.lemmatize(token)
        else:
            list1[index] = ''
    return list1

def remove_punctuation(token):
    return token.translate(None, string.punctuation)

def is_ascii(token):
    return all((ord(c) < 123)&(ord(c) > 64)  for c in token)
    
xml_to_text()
