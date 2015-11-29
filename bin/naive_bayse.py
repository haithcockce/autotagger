import sets
import math
import heapq
import preprocess_to_questions as pq
from nltk.corpus import stopwords as sw

stopwords = set(sw.words('english'))

class NaiveBayseClassifier:
  def __init__(self, clcount=1, alpha=1,  mle=False, word_frequency=True):
    self.alpha = alpha
    self.clcount = clcount
    self.mle = mle
    self.word_freq = word_frequency
  def Train(self, questions):
    if self.word_freq:
      self.TrainWordFreq(questions)
    else:
      self.TrainDocPresense(questions)

  def TrainWordFreq(self, questions):
    self.tag_counts = {}
    self.word_counts_per_tags = {}
    self.words_per_tag = {}
    
    self.all_questions = 0
    self.allwords = set()
    # Count all questions per tag.
    for question in questions:
      self.all_questions += 1
      for tag in question.tag_list:
        self.tag_counts[tag] = self.tag_counts.setdefault(tag, 0) + 1
      
      word_counts = [self.word_counts_per_tags.setdefault(tag, {}) for tag in question.tag_list]
      
      for word, count in question.word_counts.iteritems():
        self.allwords.add(word)
        for tag in question.tag_list:
          self.words_per_tag[tag] = self.words_per_tag.get(tag, 0)+count
          
        for w in word_counts:
          if word not in stopwords:
            w[word] = w.get(word, 0) + count
        
    # Convert to log probabilities
    self.log_all_questions = math.log(self.all_questions)
    log_alpha = math.log(self.alpha)
    self.uknown_uknown = log_alpha - math.log(self.alpha*len(self.allwords))
    log_total_word_counts = dict([(tag, math.log(word_count+self.alpha*len(self.allwords))) for tag, word_count in self.words_per_tag.iteritems()])

    self.unknown_word_prob_tag = dict([(tag, log_alpha - ltwc) for tag, ltwc in log_total_word_counts.iteritems()])
    self.prob_tag = dict([(tag, math.log(tag_count)-self.log_all_questions) for tag, tag_count in self.tag_counts.iteritems()])
    self.prob_word_given_tag = dict([(tag, dict([(word, math.log(cnt+self.alpha) - log_total_word_counts[tag]) for word, cnt in word_counts.iteritems()])) for tag, word_counts in self.word_counts_per_tags.iteritems()])

    
  def TrainDocPresense(self, questions):
    self.tag_counts = {}
    self.word_counts_per_tags = {}
    self.words_per_tag = {}
    
    self.all_questions = 0
    self.allwords = set()
    # Count all questions per tag.
    for question in questions:
      self.all_questions += 1
      for tag in question.tag_list:
        self.tag_counts[tag] = self.tag_counts.setdefault(tag, 0) + 1
      
      word_counts = [self.word_counts_per_tags.setdefault(tag, {}) for tag in question.tag_list]
      
      for word, count in question.word_counts.iteritems():
        self.allwords.add(word)
        for tag in question.tag_list:
          self.words_per_tag[tag] = self.words_per_tag.get(tag, 0)+count
          
        for w in word_counts:
          if w not in stopwords and count > 0:
            w[word] = w.get(word, 0) + 1
        
    # Convert to log probabilities
    self.log_all_questions = math.log(self.all_questions)
    laplace_log_all_questions = math.log(self.all_questions+self.alpha)
    log_alpha = math.log(self.alpha)
    self.uknown_uknown = log_alpha - laplace_log_all_questions

    self.unknown_word_prob_tag = dict([(tag, log_alpha - self.log_all_questions) for tag in self.words_per_tag.keys()])
    self.prob_tag = dict([(tag, math.log(tag_count)-self.log_all_questions) for tag, tag_count in self.tag_counts.iteritems()])
    laplace_doc_count = dict([(tag, math.log(tag_count+self.alpha)) for tag, tag_count in self.tag_counts.iteritems()])
    self.prob_word_given_tag = dict([(tag, dict([(word, math.log(cnt+self.alpha) - laplace_doc_count[tag]) for word, cnt in word_counts.iteritems()])) for tag, word_counts in self.word_counts_per_tags.iteritems()])

  def Classify(self, question):
    log_all_questions = math.log(self.all_questions)
    log_likelihoods = []
    
    for tag, prob_word_given_tag in self.prob_word_given_tag.iteritems():
      uknown_prob = self.unknown_word_prob_tag.get(tag, self.uknown_uknown)
      loglikelyhood = 0
      for word, count in question.word_counts.iteritems():
        if word in self.allwords:
          # log(x*y) = log(x) + log(y)
          loglikelyhood += count*prob_word_given_tag.get(word, uknown_prob)
      if not self.mle:
        loglikelyhood += self.prob_tag[tag]
      heapq.heappush(log_likelihoods, (loglikelyhood, tag))
      if len(log_likelihoods) > self.clcount:
        heapq.heappop(log_likelihoods)
    sorted_ll = reversed([heapq.heappop(log_likelihoods) for i in range(len(log_likelihoods))])
    # print sorted_ll
    return [tag for _, tag in sorted_ll]
