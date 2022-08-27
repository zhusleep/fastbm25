# Installation
The easiest way to install this package is through pip, using

```
pip install fastbm25
```
# fastbm25
The fast bm25 algorithm for text match optimized by reverted index. So the complexity will be no more than O(N(log N)).

（利用倒排索引加速的bm25文本匹配算法，从一堆数据中寻找最相似的文本）

# usage
## find top k similar sentences from corpus; Note you should tokenize text and use stop words in advance
```
from fastbm25 import fastbm25

corpus = [
    "How are you !",
    "Hello Jack! Nice to meet you!",
    "I am from China, I like math."
]
tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
model = fastbm25(tokenized_corpus)
query = "where are you from".lower().split()
result = model.top_k_sentence(query,k=1)
print(result)
```
The result is list of tuple like
> [('I am from China, I like math.', 2, -0.06000000000000001)]

For some language like Chinese that doesn't need tokenization. you can use this example
```
from fastbm25 import fastbm25

corpus = [
    "张三考上了清华",
    "李四考上了北大",
    "我烤上了地瓜.",
    "我们都有光明的未来."
]
model = fastbm25(corpus)
query = "我考上了大学"
result = model.top_k_sentence(query,k=1)
print(result)
```
> [('李四考上了北大', 1, 1.21)]
## find document pair similarity  between document a and document b
Note that a and b don't need to be included in the reference corpus;

```
from fastbm25 import fastbm25
corpus = [
    "How are you !",
    "Hello Jack! Nice to meet you!",
    "I am from China, I like math."
]
tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
model = fastbm25(tokenized_corpus)
document_a = "where are you from".lower().split()
document_b = "where are you".lower().split()

result = model.similarity_bm25(document_a,document_b)
print(result)
```
> 1.944187075527278

