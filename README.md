# What The Food

`What The Food` is an image-to-recipe application built with a React frontend and a FastAPI backend.  
Given a dish image, the backend uses a CLIP-based retrieval pipeline to identify the closest matching recipe from the dataset and returns its title, ingredients, instructions, and similarity score.

## Stack

- React + Vite frontend
- FastAPI backend
- Hugging Face `openai/clip-vit-base-patch32`
- Docker Compose for local orchestration

## Services

When the project is running, the following endpoints are available:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Running with Docker Compose

From the project root:

```bash
cd /path/to/what-the-food
```

### Development mode

Development mode runs:

- the frontend with the Vite dev server
- the backend with Uvicorn `--reload`
- the MLflow UI on port `5000`
- a model warmup service that prepares the CLIP model before backend startup

First run or after infrastructure changes:

```bash
docker compose -f docker-compose.dev.yml up --build
```

Subsequent runs:

```bash
docker compose -f docker-compose.dev.yml up -d
```

### Production-like mode

This mode serves the frontend through Nginx and runs the backend without hot reload.

First run or after infrastructure changes:

```bash
docker compose up --build
```

Subsequent runs:

```bash
docker compose up -d
```

## Runtime Notes

The project is optimized for ML development workflows:

- the CLIP model is not baked into the Docker image
- model preparation is handled by the `model-init` service
- Hugging Face artifacts are reused from the host cache at `~/.cache/huggingface`
- PyTorch is installed in CPU-only mode inside Docker

This keeps image rebuilds smaller and avoids re-downloading model artifacts on every build.

## Common Commands

Start development environment:

```bash
docker compose -f docker-compose.dev.yml up -d
```

Start production-like environment:

```bash
docker compose up -d
```

Stop the running environment:

```bash
docker compose -f docker-compose.dev.yml stop
```

Remove the development environment:

```bash
docker compose -f docker-compose.dev.yml down
```

Inspect logs:

```bash
docker compose -f docker-compose.dev.yml logs -f
```

### MLflow experiment tracking

Run the offline experiment tracking pipeline:

```bash
docker compose -f docker-compose.dev.yml exec backend sh -lc "cd /app/backend && PYTHONPATH=/app/backend python scripts/run_experiments.py"
```

This run logs:

- `model_id`
- dataset fingerprint (`SHA256`)
- text threshold
- image/text weights
- cleaned dataset statistics
- generated embeddings and run summary as artifacts

Generated outputs are stored locally in:

- `./mlruns/`
- `./artifacts/mlflow/`

Start the MLflow UI:

```bash
docker compose -f docker-compose.dev.yml up -d mlflow
```

Then open:

```text
http://localhost:5000
```

## API

Main prediction endpoint:

```http
POST /recipes/predict
```

The request must include an image file as `multipart/form-data`.

## Repository Structure

```text
what-the-food/
├── backend/
├── frontend/
├── data/
├── docker-compose.yml
└── docker-compose.dev.yml
```


## TESTS
 
Run all the backend functions tests:

```bash
docker compose -f docker-compose.dev.yml exec backend pytest -v
```

Run a single backend test file:

```bash
docker compose -f docker-compose.dev.yml exec backend pytest tests/test_health.py -v
```
```bash
docker compose -f docker-compose.dev.yml exec backend pytest tests/test_recipe_service.py -v
```
```bash
docker compose -f docker-compose.dev.yml exec backend pytest tests/test_recipes_api.py -v
```
