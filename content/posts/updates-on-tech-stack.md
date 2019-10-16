---
title: "Updates on Tech Stack"
date: 2019-10-16T12:56:47+01:00
draft: false
---

## Database

I have spent quite a bit of time experimenting with the openapi code generator and the specification to feel like I can start working on the generated Connexion boilerplate code.

I'll be using the [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) ORM through its Flask extension [flask-sqlalchemy](https://github.com/pallets/flask-sqlalchemy). I don't have much experience working with databases so this mature yet simple ORM will allow me to get started quickly. In fact I can simply things _even_ further by using the [openapi-SQLAlchemy](https://github.com/jdkandersson/openapi-SQLAlchemy) python module which, given an openapi specification automatically generates SQLAlchemy models based on those declared in the API specification, making model generation quite a "hands off" experience. From the same developers of SQLAlchemy: [Alembic](https://github.com/sqlalchemy/alembic) which, combined with its [Flask extension](https://github.com/miguelgrinberg/Flask-Migrate) allows for painless database migrations. Lastly I will also be using the [Marshmallow](https://github.com/marshmallow-code/marshmallow) serialization/deserialization library with its [SQLAlchemy extension](https://github.com/marshmallow-code/marshmallow-sqlalchemy) in order to handle database to JSON deserialization.

All in all this should allow me to have a very painless usage of databases. I say databases because I'm unsure as to which one I will be using, currently I have experience with PostgreSQL and SQLite in development because I have some familiriaty with them. I'm leaning towards PostgreSQL but will probably decide later on; that is the beauty of SQLAlchemy, it makes it relatively easy to switch database.

## Documentation

For autogenerating documentation from python code docstring I will be using [sphinx](https://github.com/sphinx-doc/sphinx) (and probably publishing the documentation to gitlab pages). I am familiar with it and I really enjoy the handsoff approach to documentation generation.
