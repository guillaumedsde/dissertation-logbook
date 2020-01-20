---
title: "Implementing Redactions"
date: 2019-11-19T17:47:10Z
draft: false
---

This week I've successfully done a couple of things:

- Dockerfile for all application services (frontend and backend)
- docker-compose to launch everything easily
- CI pipeline for building docker images
- switch to MongoDB as a datastore
- implemented sensitive section API endpoint
  - unfinished frontend implementation

## MongoDB

Why [MongoDB](https://www.mongodb.com/), especially given the praise for ElasticSearch's (ES) auto classification features that [I wrote here]({{< relref "/posts/elasticsearch-nlp-ml.md" >}})?

First, as I noted in the article above, the Lucene backed ElasticSearch classifier is not as flexible as writing my own classification code in the API with scikit-learn. ElasticSearch's auto classification at upload time would have been nice, but if I want the flexibility, I will need to also implement this in the Flask API.

Second, as [noted by the developers of ElasticSearch](https://www.elastic.co/blog/found-elasticsearch-as-nosql), using it as a NoSQL database can be done, but has drawbacks, as others [have also noted](https://www.quora.com/Why-shouldnt-I-use-ElasticSearch-as-my-primary-datastore).

I have stumbled upon some of these issues, notably ES' non atomic writes: a write in ElasticSearch is not immediatly propagated. The ES API has workarounds like [waiting for a change to be propagated](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-refresh.html) which I've used successfully, however, I also stumbled upon another issue.

I was trying to append an element to an array in an ES document's field and found [this solution](https://stackoverflow.com/questions/51384829/how-to-append-to-an-array-in-elasticsearch-using-elasticsearch-py). While ES' query language certainly is powerful, sending scripts in strings over JSON is rather inelegant in my opinion (and overly complex).

With these changes, I have quite painlessly refactored my backend API to use MongoDB as a document backend (I did use MongoDB before). MongoDB is also a 'NoSQL', schema-less database, however it functions more like a "traditional" database with atomic writes. Furthermore, its Python client is well implemented and its query language is less "scripty" and lastly, it operates with [BSON](https://en.wikipedia.org/wiki/BSON), a binary version of JSON with more flexible datatypes, convenient since I'm mostly working with JSON data.

## Redaction of sensitive sections

I have implemented the backend for storing JSON objects representing sensitive sections fairly easily, However, I am struggling to get the frontend working.

{{< figure src="/redacting/buggy_redaction.gif" caption="Current (buggy) implementation of redactions" >}}

I will try to explain my problem concisely. I am using [Mozilla's react-content-marker](https://github.com/mozilla/react-content-marker) library for highlighting text, notably with a black highlighter for redactions. This library works by placing `<mark>` HTML elements around a Regex match. I have written Regex to match character offsets in the text (in order to have a "range of redacted characters"). However, the library works by adding these tags one after the other, so the offsets given are relative to the previous tag (as can be seen in the animation above). This solution does not feel very "reliable", I'm unaware of a "best practice alternative"...

Before settling on Mozilla's solution, I tried other JS text highlighting libraries such as [react-highlight-words](https://github.com/bvaughn/react-highlight-words) which can only highlight in one color and it is not possible to "daisy chain" multiple highlighters with different colors.

I've tried exploring "all in one" React text editing libraries such as [Facebook's DraftJS](https://github.com/facebook/draft-js) but they seemed geared towards full text editing and I've not successfully disabled editing of the text in order to only keep the highlighting functionality.

This is where I currently stand on redaction feature, I will need to explore more libraries to see what is doable.
