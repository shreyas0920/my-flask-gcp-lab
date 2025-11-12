# MLOps – Flask + GCP Lab (Wine Classification)

## Original Lab
- Trains a **LogisticRegression** on the **Iris** dataset (4 features).  
- Serves predictions via a **Flask** `/predict` endpoint.  
- Simple Streamlit UI with 4 sliders.

---

## My Modifications

| Change | Why it matters |
|--------|----------------|
| **Switched to Wine Quality dataset** (13 chemical features, 3 classes) | Demonstrates handling a richer, real-world dataset. |
| **LogisticRegression + StandardScaler** (saved as `wine_model.pkl` + `wine_scaler.pkl`) | Scaling is required for convergence; model now generalises better. |
| **Full 13-feature Streamlit UI** with sensible defaults (alcohol ≈ 13, proline ≈ 700, …) | Shows end-to-end user interaction with every model input. |
| **Dockerfile + `gunicorn`** (`main:app`) | Production-ready container, works locally *and* on Cloud Run. |
| **Health-check route (`/`)** + `$PORT` handling | Avoids 404 on Cloud Run root; works with GCP’s dynamic port. |
| **GitHub repo + CI-ready structure** (`src/`, `model/`, `Dockerfile`, `requirements.txt`) | Clean, reproducible code base for future labs. |
| **Deployed to GCP Cloud Run** (`https://wine-api-590253223403.us-central1.run.app`) | Fully serverless, auto-scaling, public API. |

---

## Learnings / Challenges

| Issue | How I solved it |
|-------|-----------------|
| `ModuleNotFoundError: predict` inside Docker | Added `__init__.py` in `src/` **and** `sys.path.append(root)` in `main.py`. |
| Model not saving (`model/` empty) | Ran `train.py` from **project root**; used absolute paths in `train.py` and `predict.py`. |
| 404 when opening Cloud Run URL | Added a simple `/` health-check route. |
| GCP uses a dynamic `$PORT` | `port = int(os.environ.get("PORT", 5000))` in `main.py`. |
| `gcloud builds submit` failed | Must be in folder containing `Dockerfile`. Used `cd /d "D:\…\my-flask-gcp-lab"` first. |

---

## Prior Knowledge
- Basic **scikit-learn** (train-test split, pipelines).  
- **Flask** routing & JSON handling.  
- **Docker** basics (build, run, multi-stage).  
- **Git** workflow (init, add, commit, push).

---

## Run Instructions

### 1. Local (Docker)

```bash
# From project root
docker build -t wine-api .
docker run -p 5000:5000 wine-api