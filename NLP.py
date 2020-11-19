from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.core.window import Window
import os
import string
import re
import math
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
import numpy as np

Window.clearcolor = (1, 1, 1, 1)
Window.size = (900, 900)

# inisialisasi dan sebagainya
files = []
show_text = ""
boolean_solved_text = ""
tfidf_solved_text = ""
vsm_solved_text = ""

factory = StemmerFactory()
stemmer = factory.create_stemmer()
lines_in_file = ""

tokens_documens = []
all_files_tokens = []
doc_TF = []
doc_IDF = []

# boolean
listoperator = ['AND', 'OR', 'NOT']
listTF = []
term_total = 0

listquery = []

global_query = ""

total_term = 0


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    # def _tokenize_corpus(self, corpus):
    #     pool = Pool(cpu_count())
    #     tokenized_corpus = pool.map(self.tokenizer, corpus)
    #     return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "Dokumen yang diberikan tidak sesuai dengan indeks korpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score.tolist()

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class Root(Screen):

    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    text_query = ObjectProperty(None)
    boolean_solved_text = ObjectProperty(None)
    tfidf_solved_text = ObjectProperty(None)
    vsm_solved_text = ObjectProperty(None)
    bm35l_solved_text = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        global show_text, all_files_tokens, tokens_documens, lines_in_file
        with open(os.path.join(path, filename[0]), encoding="utf-8") as stream:
            lines_in_file = stream.read()
            files.append(lines_in_file)

            lines_in_file = lines_in_file.lower()  # jadiin lowercase
            lines_in_file = re.sub(r'\d+', '', lines_in_file)  # hapus angka
            lines_in_file = lines_in_file.translate(str.maketrans(
                '', '', string.punctuation))  # hapus tanda baca
            lines_in_file = lines_in_file.strip()  # Menghapus whitepace (karakter kosong)
            nltk_tokens = nltk.word_tokenize(lines_in_file)  # tokenize
            listStopword = set(stopwords.words(
                'indonesian'))  # inisiasi stopword

        new_lines_in_file = []
        for t in nltk_tokens:
            if t not in listStopword:
                new_lines_in_file.append(t)
                all_files_tokens.append(t)
        tokens_documens.append(new_lines_in_file)

        # array baru buat simpen mana aja yg type
        type = []
        listtype = []
        countType = 0
        for word in all_files_tokens:
            if word not in type:
                type.append(word)
            else:
                listtype.append(word)
                countType = countType + 1

        for i, sentence in enumerate(files):
            if sentence not in show_text:
                show_text += 'File {} berisi : \n'.format(i + 1)
                show_text += sentence + '\n'

        show_text += '\nTotal Tokens : {}\nTotal Type: {}\n\n'.format(
            len(all_files_tokens), countType)
        self.text_input.text = show_text
        self.dismiss_popup()

    def BooleanButton(self):
        global tokens_documens, listTF, term_total, boolean_solved_text, listoperator
        listC = []
        error = False
        # mencari frekuensi kemunculan kata pada setiap dokumen yg diinput
        for index, dokumen in enumerate(tokens_documens):
            for kata in dokumen:
                flag_token = 0
                for element in listTF:
                    if (element[0] == kata):
                        flag_token = 1
                        element[index + 1] = 1
                if (flag_token == 0):
                    listTF.append([kata])
                    for i in range(1, index + 1):
                        listTF[term_total].append(0)
                    listTF[term_total].append(1)
                    for i in range(index + 1, len(tokens_documens)):
                        listTF[term_total].append(0)
                    term_total = term_total + 1

        query = self.text_query.text
        query = query.split()
        # nyari list dokumen mana aja yg ada kata (querynya)
        for kata in query:
            # pemisahan operator
            if (kata in listoperator):
                listC.append(kata)
            # mencari kata(query) pada dokumen
            else:
                for element in listTF:
                    if (kata == element[0] and (kata not in listoperator)):
                        listC.append(element[1:])
                    else:
                        continue

        print(listC)
        temp1 = []
        temp2 = []
        operator = ''
        if (len(listC) > 1):
            while len(listC) > 0:
                if (listC[0] in listoperator):
                    operator = listC[0]
                    listC.pop(0)
                    # temp1 nyimpen array untuk kemunculan query 1 pada dokumen ke-n [1,1,0](misal)
                elif not temp1:
                    temp1 = listC[0]
                    listC.pop(0)
                    # temp2 nyimpen array untuk kemunculan query 2 pada dokumen ke-n
                elif not temp2:
                    temp2 = listC[0]
                    listC.pop(0)
                # jika menggunakan operator AND
                if (operator == 'AND' and not (not temp2)):
                    for index in range(0, len(temp1)):
                        if (temp1[index] == 1 and temp1[index] == temp2[index]):
                            temp1[index] = 1
                        else:
                            temp1[index] = 0
                    temp2.clear()
                # jika menggunakan operator OR
                elif (operator == 'OR' and not (not temp2)):
                    for index in range(0, len(temp1)):
                        if (temp1[index] == 1 or temp2[index] == 1):
                            temp1[index] = 1
                        else:
                            temp1[index] = 0
                    temp2.clear()
                # jika menggunakan operator Not
                elif (operator == 'NOT' and not (not temp2)):
                    for index in range(0, len(temp1)):
                        if (temp1[index] == 1 and temp1[index] == temp2[index]):
                            temp1[index] = 0
                        else:
                            temp1[index] = temp1[index]
                    temp2.clear()
        elif (len(listC) == 1):
            temp1 = listC[0]
        else:
            error = True

        print("temp1 : {} ".format(temp1))
        print("temp2 : {} ".format(temp2))

        if(error):
            boolean_solved_text = "Query tidak ada di dokumen manapun"
            self.boolean_solved_text.text = boolean_solved_text
        else:
            boolean_solved_text = "(Boolean) Query ada di dokumen : "
            for index, i in enumerate(temp1):
                if i == 1:
                    doc_index = index
                    boolean_solved_text += '{}, '.format(
                        doc_index + 1)
                    self.boolean_solved_text.text = boolean_solved_text
                    listC.clear()
            for index, i in enumerate(temp2):
                if i == 1 and doc_index != index:
                    boolean_solved_text += '{}, '.format(
                        index + 1)
                    self.boolean_solved_text.text = boolean_solved_text
                    listC.clear()

    def tfidfButton(self):
        global tokens_documens, listTF, term_total, tfidf_solved_text, doc_TF, doc_IDF, listquery, total_term
        tfidf_list_query = []
        tfidf()
        query = self.text_query.text
        query = query.split()
        error = False

        for kata in query:
            for element in doc_TF:
                if (kata == element[0]):
                    tfidf_list_query.append(element[1:])
                else:
                    continue

        if(len(tfidf_list_query) <= 0):
            error = True


        if (error):
            tfidf_solved_text = "Query tidak ada di dokumen manapun"
            self.tfidf_solved_text.text = tfidf_solved_text
        else:
            stack = list()

            for i in range(0, len(tokens_documens)):
                stack.append(0)

            for i in tfidf_list_query:
                for index, x in enumerate(i):
                    stack[index] = stack[index] + x

            hasil_sort = sorted(range(len(stack)),
                                key=lambda k: stack[k], reverse=True)

            tfidf_solved_text = "(TF-IDF) Query ada di dokumen : "
            for i in range(0, len(hasil_sort)):
                hasil_sort[i] = hasil_sort[i] + 1
                tfidf_solved_text += '{}, '.format(
                    hasil_sort[i])

            wij = 0
            for value in stack:
                for value2 in stack:
                    if value > value2:
                        wij = value

            self.tfidf_solved_text.text = tfidf_solved_text

    def vsmButton(self):
        global tokens_documens, listTF, term_total, vsm_solved_text, doc_TF,  doc_IDF, listquery
        vsm_list_query = listquery
        query_tf_idf = []
        error = False
        tfidf()
        query = self.text_query.text
        query = query.split()
        for kata in query:
            for index, element in enumerate(doc_TF):
                if kata == element[0]:
                    query_tf_idf.append(doc_IDF[index])
                    vsm_list_query.append(element[1:])

        cosine_similarity = list()
        for i in range(0, len(tokens_documens)):
            cosine_similarity.append(0)

        if(len(vsm_list_query) <= 0):
            error = True

        if(error):
            vsm_solved_text = "Query tidak ada di dokumen manapun"
            self.vsm_solved_text.text = vsm_solved_text
        else:
            for i in range(len(tokens_documens)):
                dot_product = 0
                pow_query = 0
                pow_doc = 0
                for count, j in enumerate(vsm_list_query):
                    dot_product = dot_product + (query_tf_idf[count] * j[i])
                    pow_query = pow_query + math.pow(query_tf_idf[count], 2)
                    pow_doc = pow_doc + math.pow(j[i], 2)
                try:
                    cosine_similarity[i] = dot_product / \
                                           (math.sqrt(pow_query) * math.sqrt(pow_doc))
                except:
                    cosine_similarity[i] = 0
            vsm_list_query.clear()
            hasil_sort = sorted(range(len(cosine_similarity)),
                                key=lambda k: cosine_similarity[k], reverse=True)
            vsm_solved_text = '(VSM) Query ada di dokumen : '
            for i in range(0, len(hasil_sort)):
                hasil_sort[i] = hasil_sort[i] + 1
                vsm_solved_text += '{}, '.format(
                    hasil_sort[i])
                self.vsm_solved_text.text = vsm_solved_text

    def BM35LButton(self):
        error = False
        value_error = 0
        bm25 = BM25L(tokens_documens)
        query = self.text_query.text
        query = query.split()
        doc_scores = bm25.get_scores(query)

        new_doc_score = []

        for value in range(len(doc_scores)):
            new_doc_score.append(doc_scores[value])

        for value in doc_scores:
            value_error = value_error + value

        if (value_error <= 0):
            error = True

        if(error):
            doc_string = "Query tidak ada di dokumen manapun"
            self.bm35l_solved_text.text = doc_string
        else:
            doc_string = "(BM25L) Query ada di dokumen : "
            hasil_sort = sorted(range(len(new_doc_score)), key=lambda k: new_doc_score[k], reverse=True)

            for i in range(0, len(new_doc_score)):
                hasil_sort[i] = hasil_sort[i] + 1
                doc_string += "{}, ".format(hasil_sort[i])

            self.bm35l_solved_text.text = doc_string

class NlpApp(App):
    def build(self):
        return Root()

def computeIDF():
    global tokens_documens, doc_TF, doc_IDF
    for element in doc_TF:
        countterm = 0
        for x in element[1:]:
            if x > 0:
                countterm = countterm + 1
        try:
            hasil = math.log(len(tokens_documens) / countterm)
        except:
            hasil = 0
        doc_IDF.append(hasil)


def computeTFIDF():
    global doc_TF, doc_IDF
    count = 0
    for element in doc_TF:
        for i in range(1, len(tokens_documens) + 1):
            element[i] = abs(element[i] * doc_IDF[count])
        count = count + 1


def tfidf():
    # mencari frekuensi kemunculan kata pada setiap dokumen yg diinput
    global tokens_documens, doc_TF, total_term
    for index, dokumen in enumerate(tokens_documens):
        for kata in dokumen:
            flag_token = 0
            for element in doc_TF:
                if (element[0] == kata):
                    flag_token = 1
                    element[index + 1] = element[index + 1] + 1
            if (flag_token == 0):
                doc_TF.append([kata])
                for i in range(1, index + 1):
                    doc_TF[total_term].append(0)
                doc_TF[total_term].append(1)
                for i in range(index + 1, len(tokens_documens)):
                    doc_TF[total_term].append(0)
                total_term = total_term + 1
    computeIDF()
    computeTFIDF()


if __name__ == "__main__":
    NlpApp().run()
