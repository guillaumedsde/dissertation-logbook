---
title: "App Stack"
date: 2019-11-01T17:06:00Z
draft: false
---

## Information Retrieval

I have tried using Terrier for Information Retrieval, I am struggling with it for reasons I will outline below.

The API I tried using for extracting documents from Terrier to try and abstract it away as a black box is still being developed, this makes it hard for me to use it. In the previous meeting, a question was raised about the app structure: how are we going to make ML perdictions using SciKit Learn in python on documents stored in Terrier (Java)? Using Two APIs is an option, the best way to do that would be to extract documents from the IR platform, process them in python, then output the final result as a rest API consumed by the React Frontend.

## Tech Stack

As said above, I have picked up my old "from scratch" OpenAPI spec and have generated as Flask project from it. The API interacts with ElasticSearch for information retrieval, its python client is fairly straightforward to use. I've adapted and migrated my old Scikit-learn code to fir the API and have gotten some endpoints running for document classification and explanation.

I have used the react material components as a starting base for my React frontend. I cannot be thankful enough to Brian Holt for his [FrontendMasters course](https://frontendmasters.com/courses/complete-react-v5/) on ReactJS, it has been **so** helpful for helping me get started with ReactJS, understanding it better and starting off with good practices.

The tech stack thus comprises of:

- **ElasticSearch** for Information retrieval
- **Flask** as a "middleware" API that fetches documents and processes them
- **ReactJS** for an interactive Javascript Frontend
