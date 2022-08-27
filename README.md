# Installation
The easiest way to install this package is through pip, using

```
pip install fastbm25
```
# fastbm25
The fast bm25 algorithm for text match optimized by reverted index. So the complexity will be no more than O(N(log N)).

（利用倒排索引加速的bm25文本匹配算法，从一堆数据中寻找最相似的文本）

# usgae
## find top k similar sentences from corpus
```
from fastbm25 import fastbm25

corpus = [
    "How are you !",
    "Hello Jack! Nice to meet you!",
    "I am from China, I like math."
]
tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
model = fastbm25(corpus)
result = model.top_k_sentence("where are you from",k=1)
print(result)
```
The result is 
> [('I am from China, I like math.', 2, -0.06000000000000001)]

