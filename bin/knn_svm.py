import sys, csv, string, copy, os, time
import question
import eval_classifier as ev
import kNN_SVMcore as core

# For compatibility with the eval module
questions = None
start_time = time.time()

def setup(args):
    global questions

    core.TRAINING_DATA_FILEPATH = args[1]

    print("DEBUG: Constructing tags and docs.")
    print(time.time() - start_time)

    core.construct_tags_and_docs()

    print("DEBUG: Constructing popular tags.")
    print(time.time() - start_time)

    core.construct_popular_tags()

    print("DEBUG: Assigning popular tags")
    print(time.time() - start_time)

    core.assign_popular_tags()

    try:
        os.mkdir('./knn_eval')
        os.mkdir('./ova_svm_eval')
    except:
        pass

    for i in range(len(core.tags)):
        questions.append(question.Question())




# KNNOVASVMClassifier
#
#

class KNNClassifier:
    svd = core.TruncatedSVD(n_components=100)
    cv = core.CountVectorizer()
    knn = core.KNeighborsClassifier(algorithm='brute')
    tfidf = core.TfidfTransformer()
    training_data = None
    eval_data = None

    def Train(self, training_data):
        pdb.set_trace()

    def Classify(self, eval_question):
        pdb.set_trace()


class OVASVMClassifier:
    svd = core.TruncatedSVD(n_components=100)
    cv = core.CountVectorizer()
    svm = core.OneVsRestClassifier(core.SVC(kernel='linear', probability=True))
    tfidf = core.TfidfTransformer()
    training_data = None
    eval_data = None

    def Train(self, training_data):
        pdb.set_trace()

    def Classify(self, eval_question):
        pdb.set_trace()


def knn_factory():
    return KNNClassifier()

def svm_factory():
    return OVASVMClassifier()


setup(sys.argv)

#ev.leave_one_out(knn_factory, ev.eval_tp1, './knn', questions, all_tags)
#ev.leave_one_out(svm_factory, ev.eval_tp1, './ova_svm', questions, all_tags)