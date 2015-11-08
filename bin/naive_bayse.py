import sets
import math
from joblib import Parallel, delayed

class NaiveBayseClassifier:
  def Train(self, questions):
    tag_counts = {}
    words_per_tag = {}
    self.all_questions = 0
    # Count all questions per tag.
    for question in questions:
      self.all_questions += 1
      for tag in question.tag_list:
        tag_counts[tag] = tag_counts.setdefault(tag, 0) + 1
      word_counters = [
        words_per_tag.setdefault(tag, {}) for tag in question.tag_list]
      for word, count in question.word_counts.iteritems():
        if count > 0:
          for counter in word_counters:
            counter[word] = counter.setdefault(word, 0)+1
    # Convert to log probabilities
    self.prob_word_given_tag = {}
    self.prob_tag = {}
    self.unknown_word_prob_tag = {}
    smoothing_questions = len(tag_counts)
    self.log_all_questions = math.log(self.all_questions+smoothing_questions)
    for tag, tag_count in tag_counts.iteritems():
      # Laplace Smoothing, one with all words
      log_tag_count = math.log(tag_count+1)
      self.prob_tag[tag] = log_tag_count - self.log_all_questions
      # Default one (Laplace smoothing)
      self.unknown_word_prob_tag[tag] = -log_tag_count
      word_counts = words_per_tag[tag]
      prob_word_given_tag = self.prob_word_given_tag.setdefault(tag, {})
      for word, count in word_counts.iteritems():
        prob_word_given_tag[word] = math.log(count+1) - log_tag_count;

  def Classify(self, question, mle=True):
    max_tag = None
    max_likelyhood = None
    log_all_questions = math.log(self.all_questions)
    
    for tag, prob_word_given_tag in self.prob_word_given_tag.iteritems():
      uknown_prob = self.unknown_word_prob_tag[tag]
      loglikelyhood = 0
      for word, count in question.word_counts.iteritems():
        if count > 0:
          # log(x*y) = log(x) + log(y)
          loglikelyhood += prob_word_given_tag.get(tag, uknown_prob)
      if mle:
        loglikelyhood += self.prob_tag[tag]
#      print 'likelyhood of {} is {}'.format(tag, loglikelyhood)
      if loglikelyhood > max_likelyhood:
        loglikelyhood = max_likelyhood
        max_tag = tag
    return max_tag

