---
title: "App Stack"
date: 2019-11-01T17:06:00Z
draft: true
---

## Information Retrieval

I have tried using Terrier for Information Retrieval, I am struggling with it for reasons I will outline below.

Firstly, the API I tried using for extracting documents from Terrier to try and abstract it away as a black box is still being developed.

This makes it hard for me to use it. Furthermore, in the previous meeting, a question was raised about the app structure: how are we going to make ML perdictions using SciKit Learn in python on documents stored in Terrier (Java)? Using Two APIs is an option, the best way to do that would be to extract documents from the IR platform, process them in python, then output the final result as a rest API consumed by the React Frontend.

Since the focus is on visualizing predictions, I think I am going to start by making a python API to be consumed by my React frontend, I will keep it simple and keep an in memory data store at first and only then worry about Elastic
