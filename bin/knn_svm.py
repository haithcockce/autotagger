import sys, csv, string, copy, os, time, pdb
import scipy
import numpy
from sklearn.utils.extmath import randomized_svd
import question
import eval_classifier as ev
import kNN_SVMcore as core

# For compatibility with the eval module
questions = []
truncated_popular_tags = None
start_time = time.time()

def setup(args):
    global questions
    global all_tags
    global truncated_popular_tags

    core.TRAINING_DATA_FILEPATH = args[1]

    #print("DEBUG: Constructing tags and docs.") py2
    #print(time.time() - start_time) py2

    core.construct_tags_and_docs()

    #print("DEBUG: Constructing popular tags.") py2
    #print(time.time() - start_time) py2

    core.construct_popular_tags()

    #print("DEBUG: Assigning popular tags") py2 
    #print(time.time() - start_time) py2

    core.assign_popular_tags()

    try:
        os.mkdir('./knn_eval')
        os.mkdir('./ova_svm_eval')
    except:
        pass


    #truncated_popular_tags = ['java','python', 'mysql']
    truncated_popular_tags = ['c++', 'angularjs', 'mysql']

    for i in range(len(core.tags)):
        if core.popular_tags[core.tags[i]] in truncated_popular_tags:
            questions.append(question.Question([core.popular_tags[core.tags[i]]], core.documents[i]))

    print "DEBUG: the amount of questions: %d" % len(questions)




# KNNOVASVMClassifier
#
#

class KNNClassifier:
    def __init__(self):
        self.svd = core.TruncatedSVD(n_components=100)
        self.cv = core.CountVectorizer()
        self.knn = core.KNeighborsClassifier(algorithm='brute')
        self.tfidf = core.TfidfTransformer()
        self.start_time = time.time()

    def Train(self, training_data):
        training_list = []
        label_list = []
        for question in training_data: 
            training_list.append(question.raw_words)
            label_list.append(core.popular_tags.index(iter(question.tag_list).next()))
        decomposed_data = self.svd.fit_transform(self.tfidf.fit_transform(self.cv.fit_transform(training_list)))
        self.knn.fit(decomposed_data, label_list)

    def Classify(self, eval_question):
        eval_doc = self.svd.transform(self.tfidf.fit_transform(self.cv.transform([eval_question.raw_words])))
        predicted_class = core.popular_tags[self.knn.predict(eval_doc)[0]]
        print "DEBUG: Classified question, knn. %f" % (time.time() - self.start_time)
        return [predicted_class]


class OVASVMClassifier:
    def __init__(self):
        self.svd = core.TruncatedSVD(n_components=100)
        self.cv = core.CountVectorizer()
        self.svm = core.OneVsRestClassifier(core.SVC(kernel='linear', probability=True))
        self.tfidf = core.TfidfTransformer()
        self.start_time = time.time()

    def Train(self, training_data):
        training_list = []
        label_list = []
        for question in training_data: 
            training_list.append(question.raw_words)
            label_list.append(core.popular_tags.index(iter(question.tag_list).next()))
        
        decomposed_data = self.svd.fit_transform(self.tfidf.fit_transform(self.cv.fit_transform(training_list)))
        self.svm.fit(decomposed_data, label_list)

    def Classify(self, eval_question):
        eval_doc = self.svd.transform(self.tfidf.fit_transform(self.cv.transform([eval_question.raw_words])))
        predicted_class = core.popular_tags[self.svm.predict(eval_doc)[0]]
        print "DEBUG: Classified question, svm. %f" % (time.time() - self.start_time)
        return [predicted_class]


def knn_factory():
    return KNNClassifier()

def svm_factory():
    return OVASVMClassifier()


setup(sys.argv)

#ev.leave_one_out(knn_factory, ev.eval_tp1, './knn_eval', questions, set(truncated_popular_tags), threads=1)
ev.leave_one_out(svm_factory, ev.eval_tp1, './ova_svm_eval', questions, set(truncated_popular_tags), threads=1)
