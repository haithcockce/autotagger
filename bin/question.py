import preprocess_to_questions as pq

def MakeQuestion(tag_list, raw_words):
    return Question(tag_list, raw_words, pq.vectorize_body(raw_words))
      
class Question:
  def __init__(self, tag_list, raw_words, word_counts, number=0):
     self.tag_list = set(tag_list)
     self.raw_words = raw_words
     self.word_counts = word_counts
     self.number = number
  

