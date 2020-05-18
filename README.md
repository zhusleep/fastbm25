# bm25
史上最快bm25倒排检索算法

# 目的
从一堆文本(corpus)中找到最相似的一句话

# 用法
```
from .bm25 import BM25
import jieba

# create model
bm25model = BM25(corpus) # corpus  like [['我'，'是'，'中国人']，['我’，'喜欢'，'苹果']]，每句话都分好词

# 函数调用
def find_top_score_index(sentence):
      """
      sentence: 待查询的句子，寻找排序高的相似问
      returns: 最相关的句子排序对应的索引，索引对应corpus真实句子
      """
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
