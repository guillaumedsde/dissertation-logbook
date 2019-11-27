---
title: "Async Hell"
date: 2019-11-27T14:15:09Z
draft: true
---

## Asynchronous pains

ReactJS's strength is that... it _reacts_ well to changes. Combined with NodeJS' flexible code capabilities, it allows for a very dynamic frontend. It is what allows me to load a document in the frontend and start redacting to only _then_ calculate a document's predicted sensitivity and load that classification along with the explanation after the fact.

My current implementation is a long HTTP `GET` request that the APi only completes when it is done calculating the prediction. This is not a great solution because from the frontend's perspective, its a very long blocking `GET` call.
