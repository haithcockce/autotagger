class Question:
  def __init__(self, tag_list, raw_words):
     self.tag_list = set(tag_list)
     self.raw_words = raw_words
  def __str__(self):
     return "{tag_list: " + self.tag_list.__str__() + ", word_count: " + self.word_counts.__str__() + "}"
