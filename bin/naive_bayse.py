import sets
import math

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
          self.words_per_tag[tag] = self.words_per_tag.get(tag, 0)+count
          
        for w in word_counts:
          w[word] = w.get(word, 0) + count
        
    # Convert to log probabilities
    self.log_all_questions = math.log(self.all_questions)
    log_alpha = math.log(alpha)
    self.uknown_uknown = log_alpha - math.log(alpha*len(self.allwords))
    log_total_word_counts = dict([(tag, math.log(word_count+alpha*len(self.allwords))) for tag, word_count in self.words_per_tag.iteritems()])

    self.unknown_word_prob_tag = dict([(tag, log_alpha - ltwc) for tag, ltwc in log_total_word_counts.iteritems()])
    self.prob_tag = dict([(tag, math.log(tag_count)-self.log_all_questions) for tag, tag_count in self.tag_counts.iteritems()])
    self.prob_word_given_tag = dict([(tag, dict([(word, math.log(cnt+alpha) - log_total_word_counts[tag]) for word, cnt in word_counts.iteritems()])) for tag, word_counts in self.word_counts_per_tags.iteritems()])

  def Classify(self, question, mle=True):
    max_tag = None
    max_likelyhood = None
    log_all_questions = math.log(self.all_questions)
    
    for tag, prob_word_given_tag in self.prob_word_given_tag.iteritems():
      uknown_prob = self.unknown_word_prob_tag.get(tag, self.uknown_uknown)
      loglikelyhood = 0
      for word, count in question.word_counts.iteritems():
        if word in self.allwords:
          # log(x*y) = log(x) + log(y)
          loglikelyhood += count*prob_word_given_tag.get(word, uknown_prob)
      if not mle:
        loglikelyhood += self.prob_tag[tag]
      if loglikelyhood > max_likelyhood:
        max_likelyhood = loglikelyhood
        max_tag = tag
    return set([max_tag])

