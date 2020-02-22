---
title: "User Study Preparations"
date: 2020-02-22T15:00:37Z
draft: false
---

## UI modifications

For the purpose of conducting the user study, I have added a `TEST_MODE` environment variable to build different variations of the frontend:

- `TEST_MODE=0`: test mode is disable no modifications to the frontend
- `TEST_MODE=1`: test mode is enabled certain "technical" features are disabled (classifier type, document split level...)
- `TEST_MODE=2`: test mode is enabled all Machine learning features are disabled (classification, explanations...)

This should allow me to display different user interfaces to different users and evaluate the effectiveness of my Machine Learning aids to the sensitivity review process.

## Selecting documents

I will be selecting a handful of documents from my data set with a couple of criteria. First I will try to find relatively short documents in so that the user study does not take too long and to simplify the process of sensitivity identification for test subjects which will not be expert reviewers.

I will also try to select documents in order to form a collection that does not particularly advantage or disadvantage each reviewer, trying to find documents that correspond to each of my reviewers' subject area.

## Order of documents

The documents in my collection share certain acronyms and words, hence, the reviewer subjects will probably spend some time on the first document trying to research and understand some of them, so they will probably spend more time on the fist document. To alleviate that, I need to either take it into account or give an introduction on a held out document and give some background information about the collection. A question I thought about: **how much detail can I give reviewers about the collection**. I could also forgo presenting details about the collection **If I could get sample redacted sensitive sections to explain the redaction process**.
