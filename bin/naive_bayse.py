import sets
import math
from joblib import Parallel, delayed

class NaiveBayseClassifier:
  def Train(self, questions, alpha=1):
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
          self.words_per_tag[tag] = self.words_per_tag.get(tag, 0)+1
          
        for w in word_counts:
          w[word] = w.get(word, 0) + 1
        
    # Convert to log probabilities
    self.prob_word_given_tag = {}
    self.prob_tag = {}
    self.unknown_word_prob_tag = {}
    self.log_all_questions = math.log(self.all_questions)
    for tag, tag_count in self.tag_counts.iteritems():
      self.prob_tag[tag] = math.log(tag_count)-self.log_all_questions
      log_total_word_count = math.log(self.words_per_tag[tag]+alpha*len(self.allwords))
      self.unknown_word_prob_tag[tag] = -log_total_word_count
      prob_word_given_tag = self.prob_word_given_tag.setdefault(tag, {})
      word_counts_per_tag = self.word_counts_per_tags[tag]
      for word in self.allwords:
        prob_word_given_tag[word] = math.log(word_counts_per_tag.get(word, 0)+alpha) - log_total_word_count
        
        

  def Classify(self, question, mle=True):
    max_tag = None
    max_likelyhood = None
    log_all_questions = math.log(self.all_questions)
    
    
    for tag, prob_word_given_tag in self.prob_word_given_tag.iteritems():
      uknown_prob = self.unknown_word_prob_tag[tag]
      loglikelyhood = 0
      for word, count in question.word_counts.iteritems():
        # log(x*y) = log(x) + log(y)
        loglikelyhood += count*prob_word_given_tag.get(word, uknown_prob)
      if not mle:
        loglikelyhood += self.prob_tag[tag]
      if loglikelyhood > max_likelyhood:
        max_likelyhood = loglikelyhood
        max_tag = tag
    return max_tag

