---
title: "Api Specification and Framework"
date: 2019-10-13T19:24:29+01:00
draft: false
---

## OpenAPI specification

[OpenAPI](https://github.com/OAI/OpenAPI-Specification) and its accompanying variety of software (most are open-source forks of a [Swagger](https://swagger.io/docs/specification/about/) equivalent) allow for very simplified API specification design.

In fact, this specification framework is so widely used that an accompaniying [openapi-generator](https://github.com/OpenAPITools/openapi-generator) (forked from swagger's) now allows for anyone, given an openapi specification to [automatically generate](https://openapi-generator.tech/docs/generators) server and/or client code in a wide variety of languages as well as documentation, schemas and configuration. For emphasis, all this **only given a single YAML file specifying an API**.

Using OpenAPI, one can define API endpoints, schemas, with their accompanying descriptions (documentation), responses, as well as indicate the types for all these Quite powerful...

I though I would be able to have gotten on quite far this week and have basic API code running, however, I feel like I should work further on the specification in order to have my first auto generated boilerplate code be as complete as possible.

So far, working from the Original Swagger 2.0 specification, I have:

- upgraded it to OpenAPI 3.0.2 standard
- fixed non restful routes
- added descriptions to already existing schemas and routes
- temporarily commented out currently unused routes
- implemented a CI pipeline for validation of the specification using an [IBM developed OpenAPI validator](https://github.com/IBM/openapi-validator) written in Node
- added deployment of the specification of the rendered YAML using [swagger-ui](https://github.com/swagger-api/swagger-ui)

This, alongside some time spent looking over the OpenAPI specfication documentation has allowed me to gain a better understanding of the OpenAPI spec itself as well as its various tools.

## Connexion

While exploring the openapi-generator I realized that while it claimed to be generating [Flask](https://github.com/pallets/flask) code, it is actually using [Connexion](https://github.com/zalando/connexion). Zalando (the shoe manufacturer) has implemented an extension to Flask to make it play more nicely with OpenAPI. It works quite similarly to Flask, all one has to do is initialize a Flask extension alongside the Flask application like so

```python
# Create the application instance
app = connexion.App(__name__, specification_dir='./')

# Read the swagger.yml file to configure the endpoints
app.add_api('swagger.yml')
```

## Database

I am relatively unexperimented with databases, however I've heard good things from the [SQLAlchemy ORM](https://github.com/sqlalchemy/sqlalchemy), I will look into it.

Another point that I stumbled upon is the storage of documents to analyze. I think in terms of scope, all I will have time for is implementing plain text and perhaps HTML documents parsing. On the other hand, PDF and Word document text extraction is not that difficult, and can potentially be implemented. If that is the case, storing the source documents will probably be better than only storing the text extracted from these documents. In fact if I do implement HTML file parsing, storing the original document might be preferable to only storing parsed text, this will enable retaining some of the formatting.

So all in all, I am currently aiming for PostgreSQL as a database which will store metadata (type, location etc...) about uploaded files (HTML, PDFs etc...) while retaining the raw original files in a directory structure echoing the structure of a document Set as I define in the API specification.


## Next

I have spent more time than expected working on the OpenAPI specification itself, but it has allowed me to better understand it, I will work on it further.

I will also work on writing more rigorous Product Requirements using the MoSCoW framework in order to better understand what endpoints I need for the API.

## References

- [Building Flask API with SQLAlchemy and connexion](https://realpython.com/flask-connexion-rest-api/)
