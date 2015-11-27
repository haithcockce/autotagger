import sys, csv, copy, time, string, pdb, os
import numpy
import scipy.sparse as spsparse
from scipy.sparse import linalg as spsplinalg
import nltk
from nltk.corpus import stopwords as sw
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# TODO 
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# ^^^^ Has a great example on creating ROC curve for each classifier

# Global variables
documents = []
tags = []
popular_tags = []
decomposed_data = []
training_data = []
validation_data = []
test_data = []
labels = []
classifier = None
svd = None
cv = None

start_time = time.time()

knn = None
svm = None


# Macros
CLASSIFICATION_METHOD = ''
OBJECTIVE = ''
TRAINING_DATA_FILEPATH = ''
TEST_DATA_FILEPATH = ''
TAGS_INDEX = 0
BODY_INDEX = 1
TRANSTABLE = {ord(c): ' ' for c in string.punctuation}



################################################################################
############################ ## HELPER FUNCTIONS ## ############################
################################################################################


# check_input
#   DESC:
#       Parses through the arguments provided to the script to make sure that
#       it is sane 
#   ARGS:
#       args: list of arguments provided to this script
#   RETN:
#       -
#   MODS:
#       -
def check_input(args):
    if len(args) < 4:
        print("Usage: python bin/kNN_SVM.py <train|validate|test> <knn|svm> <TRAINING DATA CSV> [<TEST DATA CSV>]")
        print("<RAW TEST CSV> is used for testing the knn or svm classifiers.")
        sys.exit()
    if args[1] not in ['train', 'validate', 'test']:
        print("Usage: python bin/kNN_SVM.py <train|validate|test> <knn|svm> <TRAINING DATA CSV> [<TEST DATA CSV>]")
        print("    ERROR: Expected train OR validate OR test")
        sys.exit()
    if (args[1].lower() == 'test' and len(args) < 5): 
        print("Usage: python bin/kNN_SVM.py test <knn|svm> <TRAINING DATA CSV> <TEST DATA CSV>")
        print("    ERROR: 'test' was provided, but not enough parameters was provided")
        sys.exit()
    if args[1].lower() == 'test' and not os.path.isfile(args[4]):  
        print("Usage: python bin/kNN_SVM.py test <knn|svm> <TRAINING DATA CSV> <TEST DATA CSV>")
        print("    ERROR: 'test' was provided, but could not find the TEST DATA CSV")
        sys.exit()
    if args[2].lower() not in ['knn', 'svm']: 
        print("Usage: python bin/kNN_SVM.py <train|validate|test> <knn|svm> <TRAINING DATA CSV> [<TEST DATA CSV>]")
        print("    ERROR: Expected knn OR svm")
    if not os.path.isfile(args[3]):
        print("Usage: python bin/kNN_SVM.py <train|validate|test> <knn|svm> <TRAINING DATA CSV> [<TEST DATA CSV>]")
        print("    ERROR: Could not find the TRAINING DATA CSV")
        sys.exit()


# construct_tags_and_docs
#   DESC:
#       Reads through training data and builds the documents and classes, 
#       cleaning on the way. Cleaning strips tags of their <> casing and the 
#       body of any punctuation and put to lowercase
#   ARGS:
#       -
#   RETN:
#       -
#   MODS:
#       tags: appends the ith tag set associated with the ith document. 
#       documents: appends the ith document from the corpus. 
def construct_tags_and_docs():

    # Construct the records
    csv_reader = csv.reader(open(TRAINING_DATA_FILEPATH, 'r', newline=''))
    for row in csv_reader:

        # ignore non-questions
        if row[1] != '1':
            continue

        # Ignore records that have no tags
        if len(row[16]) == 0:          
            continue 
        
        # Clean the tags and body up before appending
        cleaned_tags_buff = row[16].lower().replace('<', '').replace('>', ' ')
        cleaned_body_buff = row[8].lower().translate(TRANSTABLE)

        global tags
        global documents
        tags.append(cleaned_tags_buff)
        documents.append(cleaned_body_buff)




# construct_popular_tags
#   DESC:
#       Builds the list of most popular tags in the corpus. Most popular here is
#       defined as the tag itself occurs in more than 1% of all tag instances. 
#       1) Populate a dictionary with the vocabulary of the corpus and the 
#          frequency of each monogram. 
#       2) The counts are then normalized with L1 normalization. 
#       3) All tags with frequency lower than .01 or 1% are removed
#       4) The tags are then sorted in descending order. 
#   ARGS:
#       -
#   RETN:
#       -
#   MODS:
#       popular_tags: becomes a sorted list of tags in terms of frequency 
def construct_popular_tags():
    # Construct the dictionary of unique tags
    tag_count = {}
    for i in range(0, len(tags)):
        tag_list = tags[i].split(' ')
        for tag in tag_list:
            if len(tag) == 0:
                continue
            if tag not in tag_count:
                tag_count[tag] = 0
            else:
                tag_count[tag] += 1

    # Normalize the counts
    sigma = float(sum(list(tag_count.values())))
    normalized_tags_dict = copy.deepcopy(tag_count)
    for key in normalized_tags_dict:
        normalized_tags_dict[key] = float(normalized_tags_dict[key]) / sigma

    # Remove all tags that show up in 
    # less than 1% of all tag instances
    for i in range(0, len(normalized_tags_dict)):
        key = min(normalized_tags_dict, key=normalized_tags_dict.get)
        if normalized_tags_dict[key] < .01:
            normalized_tags_dict.pop(key, None)
        else:
            break

    # Build a sorted list of the tags in
    # order of most common to least common
    global popular_tags
    for i in range(0, len(normalized_tags_dict)):
        tag = max(normalized_tags_dict, key=normalized_tags_dict.get)
        popular_tags.append(tag)
        normalized_tags_dict.pop(tag, None)


# assign_popular_tags
#   DESC:
#       Goes through the tags and either assigns the most popular tag in the ith
#       tag set for the ith document or removes the tag set altogether along 
#       with the ith document. The motivation is to eliminate infrequent tags
#       and remove documents that do not have the tags being measured. 
#   ARGS:
#       -
#   RETN:
#       -
#   MODS:
#       tags: either the ith tag set is replaced with the most popular tag in 
#             its tag set or the ith tag set is flat out removed
#       documents: if the ith tag set is removed, then the ith document is 
#                  removed as well since it does not offer info to analysis
def assign_popular_tags():
    #pdb.set_trace()
    global tags
    checker_tags = copy.deepcopy(tags)
    for tagset in checker_tags:
        popular_tag = most_popular_tag(tagset)
        if popular_tag == None:
            i = tags.index(tagset)
            del tags[i]
            del documents[i]
        else:
            i = tags.index(tagset)
            tags[i] = popular_tags.index(popular_tag)


# most_popular_tag
#   DESC:
#       returns the most popular tag in the tag set provided. It goes through 
#       the popular_tags list that was sorted via construct_popular_tags() and
#       returns the first tag it comes across that is in both popular_tags and
#       tagset. 
#   ARGS:
#       tagset: the list of tags to check for popularity
#   RETN:
#       returns the most popular tag from tagset or None if none of the tags
#       in tagset are in popular_tags
#   MODS:
#       -
def most_popular_tag(tagset):
    for tag in popular_tags:
        if tag in tagset:
            return tag
    return None

def predict_tag(doc):
    doc_to_predict = (doc)
    decomposed = svd.transform(tfidf.fit(cv.transform(doc_to_predict)))
    label = classifier.predict(decomposed)[0]
    return popular_tags[label]
    




###############################################################################
############################### ## FUNCTIONS ## ###############################
###############################################################################

# setup
#   DESC:
#       Sets up the documents and tags for use along with other house keeping
#       things to make running easier. Does so by modifying the global variables
#   ARGS:
#       args: list of args from the start of the script (sys.argv)
#   RETN:
#       -
#   MODS: 
#       documents:  builds this by appending the body of the questions
#       tags:       builds this by appending the tag of each question
#       CLASSIFICATION_METHOD:
#                   is set to 'knn' or 'svm'
#       OBJECTIVE:  is set to 'train', 'validate', or 'test'
def setup(args):

    print("DEBUG: Checking input.")
    print(time.time() - start_time)

    check_input(args)

    global OBJECTIVE
    OBJECTIVE = args[1]
    global CLASSIFICATION_METHOD
    CLASSIFICATION_METHOD = args[2]
    global TRAINING_DATA_FILEPATH
    TRAINING_DATA_FILEPATH = args[3]
    if(OBJECTIVE == 'test'):
        global TEST_DATA_FILEPATH
        TEST_DATA_FILEPATH = args[4]

    print("DEBUG: Constructing tags and docs.")
    print(time.time() - start_time)

    construct_tags_and_docs()

    print("DEBUG: Constructing popular tags.")
    print(time.time() - start_time)

    construct_popular_tags()

    print("DEBUG: Assigning popular tags")
    print(time.time() - start_time)

    assign_popular_tags()



def train():
    global cv
    global svd
    global tfidf
    global decomposed_data
    global training_data
    global labels
    global classifier

    print("DEBUG: Count vectorizing the monograms.")
    print(time.time() - start_time)

    cv = CountVectorizer()
    monogram_frequency_matrix = cv.fit_transform(documents)

    print("DEBUG: if-idfing the monograms.")
    print(time.time() - start_time)

    tfidf = TfidfTransformer()
    tfidf_normalized_matrix = tfidf.fit_transform(monogram_frequency_matrix)

    print("DEBUG: SVDing the tfidf-ed monograms.")
    print(time.time() - start_time)

    #pdb.set_trace()

    svd = TruncatedSVD(n_components=5000)
    decomposed_data = svd.fit_transform(tfidf_normalized_matrix)

    # Now to take the normalize the columns since the eigenvalues were so heavily skewed. 
    print("DEBUG: generated decomposed_data.")
    print(time.time() - start_time)

    #global singular_values
    #singular_values = spsparse.linalg.svds(tfidf_normalized_matrix, k=10000, return_singular_vectors=False)

    training_data = decomposed_data
    labels = tags

    if CLASSIFICATION_METHOD == 'knn':
        # Defaults to 5 neighbors. bruteforce algorithm simply
        # gets the distance from a point to all other points. 
        classifier = KNeighborsClassifier(algorithm='brute')
    else:
        classifier = OneVsRestClassifier(
                        SVC(kernel='linear', probability=True)
                     )

    classifier.fit(training_data, labels)
    
    print("DEBUG: Trained classifier.")
    print(time.time() - start_time)




def validate():
    print("METHOD STUB. validate() needs to be implemented.")

def test():
    print("METHOD STUB. test() needs to be implemented.")





setup(sys.argv)
if OBJECTIVE == 'train':
    train()
elif OBJECTIVE == 'validate':
    validate()
else:
    test()
