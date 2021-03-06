FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime as requirements-gen

ENV POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    REQUIREMENTS_DIR="/opt/pysetup"


RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR $REQUIREMENTS_DIR
COPY poetry.lock pyproject.toml ./
RUN poetry export -o requirements.txt --without-hashes


FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime as production
ENV REQUIREMENTS_DIR="/opt/pysetup"

COPY --from=requirements-gen $REQUIREMENTS_DIR/requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
ENV PYTHONPATH /app

COPY ./secret_sauce ./secret_sauce
ENTRYPOINT ["python", "./secret_sauce/main.py"]