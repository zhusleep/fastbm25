# bm25
史上最快bm25倒排检索算法

# 用法
```
from .bm25 import BM25
bm25model = BM25(corpus) # corpus is like [['我'，'是'，'中国人']，['我’，'喜欢'，'苹果']]

def find_top_score_index(sentence):
      """寻找排序高的相似问"""
      score_overall = {}
      for word in jieba.cut(sentence):
          if word not in bm25model.document_score:
              continue
          for key, value in bm25model.document_score[word].items():
              if key not in score_overall:
                  # print(score_overall)
                  score_overall[key] = value
              else:
                  score_overall[key] += value
      if score_overall:
          return sorted(score_overall.items(), key=lambda x: x[1], reverse=True)
      else:
          return None
```
