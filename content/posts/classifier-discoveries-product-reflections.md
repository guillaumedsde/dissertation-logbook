---
title: "Classifier Discoveries Product Reflections"
date: 2019-10-01T11:14:09+01:00
draft: false
---

# Classifier discoveries, NLP frameworks and reflections on product

## Classifier "improvements"?
I have perfected the classifier further, notably using multiple processes in order to leverage multiple cores while training the models. There is plenty of optimization to be done so I stoppped here. In terms of text pre-processing, I have changed my pipeline to:

* remove english stopwords
* lowercase all text
* strip accents

One thing I have noted is that [Lime](https://github.com/marcotcr/lime) still returns some words with uppercase letters, which makes me wonder if all this text pre-processing is being applied by Lime or not. My guess is that it is not, because I am using a scikit `Pipeline` rather than a `Classifier`, so in order to use Lime, I might have to split my pipeline into distinct steps.


## Discoveries
In the hopes of obtaining better results. I have come to realizer a couple of things.

Firstly, I have no established test, so I am "optimizing" at random without knowing whether these changes are effective. I would need a test suite to determine the effectiveness of these changes.

Secondly, there is no point using the scikit predefined classifier test comparison since I will not be categorizing texts according to the 20 newsgroup data categories, rather I will be classifying documents as either sensitive/non sensitive.

Thirdly, using [Lime](https://github.com/marcotcr/lime), I have come to realize that the features highlighted by Lime which result in the classification of the document will not necessarily be the sensitive extracts from the document. For example, a document mentionning the names of GCHQ surveillance programs might contain sensitive information, but this sensitive informaiton might be the names of people associated with the programs rather than the programs themselves. As such using a binary classification of sensitive/non-sensitive might result in Lime highlighting *why* the document is sensitive as opposed to *what* in the document is sensitive.

## Text processing and more classification testing
I have downloaded a copy of the redacted Snowden documents and have written a python script to extract text data from these documents (PDFs only for now). For this, I have used a Python package called [textract](https://github.com/deanmalmgren/textract) which interestingly is also capable of extracting text from a variaety of other document types, this might be useful later.

I have ran them through my classifier and have obtained acceptable results with the majority of classifiers categorizing them as either `sci.crypt` (science - cryptography) or `comp.os` (computing science - operating systems). Again, keeping in mind, this is mostly for learning how to manipulate scikit more than to produce anything relevant at this stage. I think I will dive into the specifics of each classifier once I have tested them on actual data rather than trying to understand them beforehand.

## Reflections on the product
While extracting text from PDFs, I have come to realize an important point. When I initially printed out the extracted text from the document, my enthusiasm for the success of the text extraction made me blind to an important issue: While extracting text, I lose all formatting, and images in the document. Which in turn made me realize; while redacting *raw text* is not very hard, redacting a document while keeping its structure is quite difficult. I initally thought I could work on various document types (.odf .pdf .png etc...) I now realize that quite a bit has to be done in order to implement redaction for each filetype **and** keep formatting and any non-textual data.

I have also, as part of my summer progress report, formulated a handful of user stories for the final product, I will probably make an entire post about them, or store them elsewhere, but here they are:

* As a redactor I want my file uploaded for redaction to be pre-processed and displayed with **redaction hints** so that I can redact documents faster
* As a redactor, I expect my redactions of non-dictionary words to be **reflected across an entire document** so that I don't have to go over them again
* As a redactor, I would like the suggested redactions to **improve as I redact documents**
* As a redactor, I would like to redact on my MacOS, Windows or Linux based computer so that I don't have to worry about the device I am using
* As a redactor, I want to add and remove redactions so that I can go back and **edit** my work.
* As a redactor, I want the files I redact to **not be retained as a whole** so that I can keep them confidential

## Useful frameworks
While browsing around, I stumbled upon a python library called [spaCy](https://github.com/explosion/spaCy). A mature Natural Language Processing framework that offers many features, notably "Named entity recognition" and "Built in visualizers for syntax and NER".

Another potentially useful framework I found is [textBlob](https://textblob.readthedocs.io/en/dev/) a user friendly interface to [NTLK](https://www.nltk.org/) one of the oldest Natural Language Processing Framework.

I will try out these packages, see what I can do with them.


## References

* [Lime](https://github.com/marcotcr/lime) Explaining Machine Learning Predictions
* [textract](https://github.com/deanmalmgren/textract) Python package to extract text from various filetypes
* [spaCy](https://github.com/explosion/spaCy) a widely used Natural Language Processing framework
* [textBlob](https://textblob.readthedocs.io/en/dev/) an interface to NTLK
* [NTLK](https://www.nltk.org/) one of the oldest Natural Language Processing frameworks
