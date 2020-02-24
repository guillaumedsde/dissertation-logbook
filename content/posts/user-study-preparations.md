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

Furthermore I've enabled sensitive explanations to be displayed by default, I am also considering showing only half the maximum explanations by default to see how reviewers modify it (increasing or decreasing it).

## Fixing the classifier

In my previous attempts, I have tried to address the problem of imbalance in my dataset (many more insensitive than sensitive documents) with Stratified K-fold Cross Validation (which only replicates the "same" imbalance over K folds) as well as various [resampling strategies](https://dissertation.guillaume.desusanne.com/posts/one-hot-oversampling-and-evaluation/) which have not yielded good results.

I have managed to fix this, here is a sample from the buggy document parsing code:

```python
pool = Pool(processes=PROCESSES)
for file_path in file_paths:
    pool.apply_async(
        func=read_file,
        args=(file_path,),
        callback=texts.append,
        error_callback=logging.exception,
    )
texts = pool.map(func=read_file, iterable=file_paths)
pool.close()
pool.join()
```

critically, I use `pool.apply_async` to schedule file reads to the process pool. However, as stated by the [documentation](https://docs.python.org/3.8/library/multiprocessing.html#multiprocessing.pool.Pool.apply_async)

> If callback is specified then it should be a callable which accepts a single argument. When the result becomes ready callback is applied to it, that is unless the call failed, in which case the error_callback is applied instead.

Thus, `texts.append` is called for a text whenever it is done being parsed, thus appending text to the `text` list in arbitrary order. The fix is to use the slower `pool.map` method which preserves the ordering:

```python
pool = Pool(processes=PROCESSES)
texts = pool.map(func=read_file, iterable=file_paths)
pool.close()
```

I will re run a GridSearch with this fix and will combine a number of resampling techniques with Stratified K-folds. I also tried passing a `joblib.Memory` object to my Pipeline to [cache vectorized data](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to avoid having to re-run the vectorizer and speed things up but I experienced poor performance and did not have time to elaborate and ran the grid search without.

## Selecting documents

I will be selecting a handful of documents from my data set with a couple of criteria. First I will try to find relatively short documents in so that the user study does not take too long and to simplify the process of sensitivity identification for test subjects which will not be expert reviewers.

I will also try to select documents in order to form a collection that does not particularly advantage or disadvantage each reviewer, trying to find documents that correspond to each of my reviewers' subject area.

## Order of documents

The documents in my collection share certain acronyms and words, hence, the reviewer subjects will probably spend some time on the first document trying to research and understand some of them, so they will probably spend more time on the fist document. To alleviate that, I need to either take it into account or give an introduction on a held out document and give some background information about the collection. A question I thought about: **how much detail can I give reviewers about the collection**. I could also forgo presenting details about the collection **If I could get sample redacted sensitive sections to explain the redaction process**.

## Questionnaire

I will also give a [short questionnaire](https://docs.google.com/forms/d/1ZpeUlWykRoVqsDzYZ2s45o4u08EXxLr44hlthCFYwWk) (please don't fill out unless you have used Harpocrates) to fill out after using the application. It has questions about specific components in the interface as well as more general questions about the experience of using Harpocrates.
