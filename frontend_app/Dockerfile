# Stage 1: Build
FROM python:3.10-buster as builder

RUN pip install poetry==1.6.1

WORKDIR /frontend_app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project true && poetry install --no-root

# Stage 2: Runtime
FROM python:3.10-slim-buster

WORKDIR /frontend_app

COPY --from=builder /frontend_app .

ENV PATH="/frontend_app/.venv/bin:$PATH"

COPY streamlit_app.py .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit_app.py"]