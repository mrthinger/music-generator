FROM deepspeed/deepspeed:latest as dev

ENV HOME=/home/deepspeed

ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    REQUIREMENTS_DIR="$HOME/pysetup" \
    PATH="$HOME/.poetry/bin:$PATH"

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

FROM dev as requirements-gen

RUN mkdir -p $REQUIREMENTS_DIR
WORKDIR $REQUIREMENTS_DIR
COPY --chown=deepspeed:deepspeed poetry.lock pyproject.toml requirements.txt ./
RUN poetry export -o extra_requirements.txt --without-hashes \
    && awk '{print}' requirements.txt extra_requirements.txt > total_requirements.txt \
    && rm requirements.txt extra_requirements.txt \
    && mv ./total_requirements.txt ./requirements.txt


FROM deepspeed/deepspeed:latest as production
ENV HOME=/home/deepspeed
ENV REQUIREMENTS_DIR="$HOME/pysetup"
ENV APP_DIR="$HOME/app"

COPY --chown=deepspeed:deepspeed --from=requirements-gen $REQUIREMENTS_DIR/requirements.txt .
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir -p $REQUIREMENTS_DIR
WORKDIR $APP_DIR
ENV PYTHONPATH $APP_DIR
ENV PATH="$HOME/.local/bin:$PATH"

COPY --chown=deepspeed:deepspeed ./secret_sauce ./secret_sauce
# ENTRYPOINT ["python", "./secret_sauce/main.py"]