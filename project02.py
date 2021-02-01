import matplotlib.pyplot as plt
import nltk
import math
from nltk.tokenize import word_tokenize, sent_tokenize
from wordcloud import WordCloud
from nltk.corpus import stopwords
stopwords_list = set(stopwords.words('turkish'))
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, KneserNeyInterpolated
from nltk.tokenize.treebank import TreebankWordDetokenizer
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import numpy as np


def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    if(bagOfWordsCount == 0):
        return tfDict
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents,uniqueWords):
    N = len(documents)
    idfDict = dict.fromkeys(uniqueWords, 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def getBagOfWords(Docs, stopwords):
    bagOfWords = []
    for i in range(len(Docs)):
        doc = Docs[i]
        doc_temp = []
        for each in word_tokenize(doc):
            each = each.lower()
            if (not each.isalpha()) or (stopwords and each in stopwords_list) or (stopwords and each == "bir") or (stopwords and each == "nin"):
                continue
            doc_temp.append(each)
        bagOfWords.append(doc_temp)
    return bagOfWords

def create_WordCloud(Docs,dim_size,wordcloud_outputfile,mode="TF",stopwords=False):
    bagOfWords = getBagOfWords(Docs, stopwords)
    uniqueWords = set().union(*bagOfWords)

    numOfWords = []
    for i in range(len(Docs)):
        numOfWords_temp = dict.fromkeys(bagOfWords[i], 0)
        for word in bagOfWords[i]:
            numOfWords_temp[word] += 1
        numOfWords.append(numOfWords_temp)


    tfList = []
    for i in range(len(Docs)):
        tfList.append(computeTF(numOfWords[i], bagOfWords[i]))


    tf_sum = dict.fromkeys(uniqueWords, 0)
    for i in range(len(bagOfWords)):
        current_tf = tfList[i]
        for word, val in current_tf.items():
            tf_sum[word] += val

    wc_dict = tf_sum

    if(mode == "TFIDF"):
        idfs = computeIDF(numOfWords, uniqueWords)

        tfidfList =[]
        for i in range(len(bagOfWords)):
            tfidfList.append(computeTFIDF(tfList[i],idfs))
        
        tfidf_sum = dict.fromkeys(uniqueWords, 0)
        for i in range(len(bagOfWords)):
            current_tfidf = tfidfList[i]
            for word, val in current_tfidf.items():
                tfidf_sum[word] += val
        
        wc_dict = tfidf_sum

    wordcloud = WordCloud(width=1600,height=1600, random_state= 500).generate_from_frequencies(wc_dict)
    plt.figure(figsize=(dim_size, dim_size), facecolor='k')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.tight_layout(pad=0)
    plt.axis("off")
    plt.savefig(wordcloud_outputfile, facecolor='k', bbox_inches='tight')
    plt.clf()

def create_ZiphsPlot(Docs,zips_outputfile):
    # Docs=[]
    # Docs.append("Son dakika haberine göre Sağlık Bakanı Fahrettin Koca, Türkiye'nin günlük koronavirüs tablosunu açıkladı. Türkiye'de son 24 saatte 156 bin 642 Kovid-19 testi yapıldı, 5 bin 103 kişiye hastalık tanısı konuldu, 141 kişi hayatını kaybetti.")
    # Docs.append("Ağır hasta sayısı 3 bin 990 oldu, son 24 saatte 3 bin 19 kişinin Kovid-19 tedavisinin tamamlanmasıyla iyileşenlerin sayısı 367 bin 592'ye yükseldi. Bakan Koca yaptığı değerlendirmede,  Birlikte tedbirlere uyarsak kendi kısıtlamalarımızı kendimiz koyarsak mücadele daha kolay olacak. dedi.")
    allWords = []
    for i in range(len(Docs)):
        doc = Docs[i]
        for each in word_tokenize(doc):
            each = each.lower()
            if (not each.isalpha()):
                continue
            allWords.append(each)

    from collections import Counter
    counts = Counter(allWords)
    # print(len(counts))
    most_common = counts.most_common(len(counts))
    most_common_index = []
    most_common_count = []
    
    for i in range(len(most_common)):
        most_common_index.append(i+1)
        most_common_count.append(most_common[i][1])

    # plt.scatter(most_common_index, most_common_count, marker='o')
    # plt.grid(True)
    plt.xlabel('Word Rank')
    plt.ylabel('Word Frequency')

    most_common_index = np.log(most_common_index)
    most_common_count = np.log(most_common_count)

    # plt.tight_layout()
    plt.plot(most_common_index, most_common_count, "ro")

    plt.savefig(zips_outputfile, bbox_inches = "tight")
    plt.clf()

def create_HeapsPlot(Docs,heaps_outputfile):
    heaps = []
    uniqueWords = set()
    counter = 0
    for i in range(len(Docs)):
        doc = Docs[i]
        for each in word_tokenize(doc):
            counter += 1
            each = each.lower()
            if (not each.isalpha()):
                continue
            uniqueWords.add(each)
            heaps.append(len(uniqueWords))

    word_cnt = []
    vocab_cnt = []

    for i in range(len(heaps)):
        word_cnt.append(i)
        vocab_cnt.append(heaps[i])


    # plt.tight_layout()
    plt.plot(word_cnt, vocab_cnt)
    # plt.xscale("linear")
    plt.xlabel('Words Size')
    plt.ylabel('Vocabulary Size')


    plt.savefig(heaps_outputfile, bbox_inches = "tight")
    plt.clf()

def create_LanguageModel(Docs,model_type,ngram):
    text = " ".join(Docs)
    text = text.replace("\\n"," ")
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]
    train_data, padded_sents = padded_everygram_pipeline(ngram, tokenized_text)
    model = MLE(ngram)
    if model_type != "MLE":
        model = KneserNeyInterpolated(ngram) 
    model.fit(train_data, padded_sents)
    return model

detokenize = TreebankWordDetokenizer().detokenize
def generate_sent(LM3_MLE, text_seed):
    content = []
    while True:
        new_generated = LM3_MLE.generate(1, text_seed=text_seed)
        if new_generated == '<s>':
            continue
        if new_generated == '</s>':
            break
        content.append(new_generated)
        text_seed.append(new_generated)
        # print(text_seed)
    return detokenize(content)

def generate_sentence(LM3_MLE,text):
    min_per = 10000000000000000000000
    min_text =""
    for i in range(5):
        starting_text = ["<s>"]
        starting_text.append(text)
        generated = generate_sent(LM3_MLE, starting_text)
        test_tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(generated)]
        test_data, _ = padded_everygram_pipeline(LM3_MLE.order, test_tokenized_text)

        sentences =[]
        for test in test_data:
            for each in list(test):
                sentences.append(each)

        ngram_list =[]
        for each in sentences:
            if(len(each) == LM3_MLE.order and (each[0] != "<s>" and each[-1] != "</s>")):
                ngram_list.append(each)
        
        if(len(ngram_list)>0):
            if(LM3_MLE.perplexity(ngram_list) < min_per):
                min_per = LM3_MLE.perplexity(ngram_list)
                min_text = generated
            elif(LM3_MLE.perplexity(ngram_list) == min_per and len(generated) > len(min_text)):
                min_per = LM3_MLE.perplexity(ngram_list)
                min_text = generated
    return text+" "+min_text,min_per

def create_WordVectors(Docs,dim_size,model_type,window_size):
    # # Docs=Docs[:500]
    # Docs = []
    # Docs.append("Son dakika haberine göre Sağlık Bakanı Fahrettin Koca, Türkiye'nin günlük koronavirüs tablosunu açıkladı. Türkiye'de son 24 saatte 156 bin 642 Kovid-19 testi yapıldı, 5 bin 103 kişiye hastalık tanısı konuldu, 141 kişi hayatını kaybetti.")
    # Docs.append("Ağır hasta sayısı 3 bin 990 oldu, son 24 saatte 3 bin 19 kişinin Kovid-19 tedavisinin tamamlanmasıyla iyileşenlerin sayısı 367 bin 592'ye yükseldi. Bakan Koca yaptığı değerlendirmede,  Birlikte tedbirlere uyarsak kendi kısıtlamalarımızı kendimiz koyarsak mücadele daha kolay olacak. dedi.")
    
    text = " ".join(Docs)
    text = text.replace("\\n"," ")
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]
    

    without_stopwords = []
    for i in range(len(tokenized_text)):
        temp = []
        for each in tokenized_text[i]:
            if (each not in stopwords_list):
                temp.append(each)
        without_stopwords.append(temp)

    tokenized_text = without_stopwords

    # CBOW(0) or skip gram(1)
    model = Word2Vec(sentences=tokenized_text, size=dim_size, window=window_size, min_count=5, workers=multiprocessing.cpu_count(), sg = 1)
    if model_type == "cbow":
        model = Word2Vec(sentences=tokenized_text, size=dim_size, window=window_size, min_count=5, workers=multiprocessing.cpu_count(), sg = 0)

    return model

def use_WordRelationship(WE,example_tuple_list,example_tuple_test):
    model = WE
    words=list(model.wv.vocab)
    cleaned_tuple_list = []

    for i in range(len(example_tuple_list)):
        first = example_tuple_list[i][0]
        second = example_tuple_list[i][1]
        if first in words and second in words:
            cleaned_tuple_list.append(example_tuple_list[i])

    given_pos = 0

    if example_tuple_test[0] != '':
        given_pos = 0
    else:
        given_pos = 1
    
    if(len(cleaned_tuple_list) != 0 and (example_tuple_test[given_pos] in words)):
        difference = 0
        for i in range(len(cleaned_tuple_list)):
            first = cleaned_tuple_list[i][0]
            second = cleaned_tuple_list[i][1]
            temp = model.wv[cleaned_tuple_list[i][0]] - model.wv[cleaned_tuple_list[i][1]]
            difference += temp

        difference = difference/len(cleaned_tuple_list)
        deneme = difference
        if(given_pos == 0):
            deneme = model.wv[example_tuple_test[0]] - difference
        if(given_pos == 1):
            deneme = model.wv[example_tuple_test[1]] + difference

        out_list = []
        [out_list.append(each) for each in model.wv.similar_by_vector(deneme)]
        # print(cleaned_tuple_list)
        i = 0
        j = 5
        output_list = []
        while i<j:
            if out_list[i][0] != example_tuple_test[given_pos]:
                output_list.append(out_list[i])
            else:
                j += 1
            i += 1
        print(output_list)
    else:
        print("Sorry, this operation cannot be performed!")

