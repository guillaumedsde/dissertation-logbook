---
title: "Repositories Product and Tech stack"
date: 2019-09-24T17:18:09+01:00
draft: false
---

# Repositories Product and Tech stack

## Repositories

I have created a [gitlab group](https://gitlab.com/visualising-sensitivity-classification-features) for all project related to my level 4 dissertation. GitLab will be useful for automation through its CI/CD pipelines. I have setup a [repository](https://gitlab.com/visualising-sensitivity-classification-features/logbook) for this statically generated site in order to autobuild and publish its version controlled source.

I have also setup a [notech](https://gitlab.com/visualising-sensitivity-classification-features/notech) repostiroy for any file that is not directly source code as well as issues that do not directly relate to source code.

## Product

This [video](https://www.youtube.com/watch?v=hLgwEs1KCdQ) gives an idea of the kind of product we're looking for.

In terms of scope, what I imagine is a platform where the document is uploaded though the frontend, processed by the backend, redacted by the user and redownloaded once the redacting is done.

## Tech stack

### Backend

I understand that extracting features and sensitivites from the texts to redact will be done with python, as such, and considering my knowledge in the language, the backend probably ought the be written with it. This will probably consist in writing an API that will be consumed by a frontend. There are multiple frameworks, for this, in the past I have used Flask, which is a *barebones* approach, that is, lots of hand coding. I might look into other python API frameworks...

### Datastore

As I said above, from what I understand about the final product, storing documents might be out of scope. As such, the datastore we would potentially need is one to store the trained model for text feature extraction, but not for storing documents themselves.

### Frontend

I have experience with the django framework, but it not a fully fledged javascript framework which will probably be needed in order to have all the interactivity of the redaction process available in the frontend. I have some experience with Angular and typescript, but I have not grasped the project structure imposed by the angular framework, as such, I'm interested in looking into React as a frontend framework and using pure javascript instead of typescript.
