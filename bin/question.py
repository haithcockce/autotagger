class Question:
  def __init__(self, tag_list, word_counts):
     self.tag_list = tag_list
     self.word_counts = word_counts
  def __str__(self):
     return "{tag_list: " + self.tag_list.__str__() + ", word_count: " + self.word_counts.__str__() + "}"
