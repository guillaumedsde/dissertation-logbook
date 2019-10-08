---
title: "Application Requirements"
date: 2019-10-08T15:46:50+01:00
draft: false
---

_I have started to define Requirements for the final application. I have posted below a copy of the original requirements file, it will however evolve over time, as such, **[an always up to date version can be found here](https://visualising-sensitivity-classification-features.gitlab.io/notech/requirements.md.pdf)**._

## Purpose of the application

I am looking to build a **web application** that improves the experience of **redacting sensitive documents**. In addition to **highlight to redact** features, I want this application to learn from the document redaction process in order to **suggest a sensitivity classification** and potentially **suggest pre-redacttions** of sensitive elements within the document.

## Development

I forsee a couple _key stages_ in the development process for such an application:

- Build **Machine Learning model** for senstive document clasification and "pre-redactions"
- Connect this model to an **API** with a precise **specification**
- build **frontend** for this API
- Implement **learning** from the document redactions

## User stories and features

_Some of these are taken from [my previous work](https://dissertation.guillaume.desusanne.com/posts/classifier-discoveries-product-reflections/) on user stories for this application_

Here is a list of user stories for the application, I have extracted key features from the stories in bullet points below them.

- As a redactor I want my file uploaded for redaction to be pre-processed and displayed with **redaction hints** so that I can redact documents faster
  - file upload
  - text extraction
  - Machine Learning model for sensitivity classification
- As a redactor, I would like to **understand the reasons for the sensitivity classfication** suggestions so that I can decide on their relevance
  - Machine learning model explainer
- As a redactor, I would like the suggested redactions to **improve as I redact documents**
  - Learning loopback from user redactions to model
- As a redactor, I expect some of my redactions to be **reflected across an entire document** so that I don't have to go over them again
  - redaction assistance
- As a redactor, I would like to redact on my MacOS, Windows or Linux based computer so that I don't have to worry about the device I am using
  - web application
- As a redactor, I want to add and remove redactions so that I can go back and **edit** my work.
  - save state of current document
- As a redactor, I want the files I redact to **not be retained as a whole** so that I can keep them confidential
  - train model but discard document
- As a developer I want to standardize API calls so that others can easily built upon or improve the final application
  - API specification
