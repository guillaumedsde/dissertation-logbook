---
title: "ElasticSearch built in Natural Language Processing and Machine Learning"
date: 2019-11-06T09:45:32Z
draft: true
---

While studying [ElasticSearch](https://www.elastic.co/products/elasticsearch), I stumbled upon a [blog post](https://www.elastic.co/blog/text-classification-made-easy-with-elasticsearch) detailing how ElasticSearch can perform Natural Language Processing (NLP) and Machine Learning (ML) classification out of the box, processing documents while _indexing_ them. While experimenting exposing [Scikit Learn](https://scikit-learn.org/) predictions over a Flask API, I stumbled upon a problem: it is **slow**. Indeed, classifying a document and calculating an explanation for that classification is long and computationally expensive. Doing it while indexing the document allows us to only do it once and store the results, this is especially convenient since the document does not change.

## NLP

As stated by the blog post mentionned above, Elastic Search can perform most NLP steps at index time, namely:

- Tokenization
- Normalizing (stemming, lemmatization)
- Stopword removal

Thus when opening the document, the loading time in the fontend will b drastically reduced. We can reduce it even further by performing the ML classification at index time in Elasticsearch as well.

## ML classification

Again drawing from the blog post, it seems like ElasticSearch can perform ML classification while indexing the document.

What if the ML model changes? I need to research this further, but my intuition tells me ElasticSearch handles the reclassification of the document when the model changes. Another problem that arises is: how can we use Lime on the ElasticSearch classifier? The answer seems to lie in the `README.md` of Lime:

> All we require is that the classifier implements a function that takes in raw text or a numpy array and outputs a probability for each class. Support for [scikit-learn](https://scikit-learn.org/) classifiers is built-in.

[README.md for the Lime explainer](https://github.com/marcotcr/lime#lime)

I need to look at the code for doing this but it seems like I can make Lime explain ElasticSearch's classifications.

## Scale?

Ultimately, ElasticSearch is built upon [Lucene](https://lucene.apache.org/): it does not reinvent the wheel. All of this is "standard" ML text classification, what ElasticSearch brings to the table is an _indsutrialization_ of that process. The post articulates that very well:

> [...] tools developed at research departments are usually not very useful for an enterprise context. Very often the data formats are proprietary, the tools need to be compiled and executed on the command line, and the results are very often simply piped to standard out. REST APIs are an exception.
>
> With the Elasticsearch language analyzers, on the other hand, you only need to configure your mapping and index the data. The pre-processing happens automatically at index time.

By using ElasticSearch's NLP and ML tools (again "just" Lucene in the background) I can avoid having to write a lot of boilerplate code. I am not "losing" anything, _so long_ as I can get Lime to explain ElasticSearch's predictions.

> Why then use Elasticsearch when there are other tools?
>
> Because your data is already there and it's going to pre-compute the underlying statistics anyway. It's almost like you get some NLP for free!

## Resources

- [ML in production with ElasticSearch](https://www.elastic.co/blog/sizing-machine-learning-with-elasticsearch)
- [Getting started with ElasticSearch NLP and ML](https://dataconomy.com/tag/use-elasticsearch-nlp-text-mining/)
