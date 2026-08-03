# MLOps Capstone 2 — Casting Defect Detection

Skeleton for the **Casting Defect Detection** Deep-Learning MLOps pipeline (ResNet18 transfer
learning). Infrastructure + the test harness are provided; **you implement the modelling + MLOps
logic** marked `# TODO`. Each notebook section's marks (red) match `MLOps2_Grading_Rubric.xlsx`.

## Provided (don't rewrite)
`config.py` · `src/__init__.py` · `src/dataset.py` · `requirements.txt` · `Dockerfile` ·
`.github/workflows/ci.yml` · `tests/`

## You build (the `# TODO`s)
| File | Stage | You implement |
|---|---|---|
| `Data_Preparation.ipynb` | 1, 2.1 | guided notebook: understanding, quality, versioning, EDA, preprocessing |
| `Model_Development_and_Tracking.ipynb` | 2.2–2.4, 3 | guided notebook: transfer learning, MLflow, eval, registry, FastAPI/Docker/CI |
| `Operations_Monitoring_and_Evidence.ipynb` | 4 | guided notebook: logging, statistical + embedding drift, retraining, rollback |
| `src/data_prep.py` | 1, 2.1 | discovery, quality validation, versioned splits, transforms, image features |
| `src/model.py` | 2.2, 4.3 | ResNet18 transfer-learning model + embedding extractor |
| `src/train.py` | 2.3–2.4, 3.2 | training loop, MLflow tracking + registry + @production, reference baseline |
| `src/evaluate.py` | 3.1 | recall-on-defect / F1 / ROC / confusion + failure cases |
| `app.py` | 3.3 | FastAPI `/health` + `/predict` (image upload) + prediction logging |
| `src/monitoring.py` | 4.1–4.3 | statistical (Evidently+PSI) + embedding drift + confidence |
| `src/retrain.py` | 4.4–4.5 | drift-triggered retrain, version compare, promote/rollback |

The three notebooks carry **descriptive per-task instructions** (objective · stage/task · inputs→outputs
· TODO expectations · dependencies · where to document). In addition, each stage now opens with a
**Markdown sub-task checklist** — every sub-task shows its ID and marks (e.g. `2.2.1 — ImageNet-pretrained
ResNet18, backbone frozen [4]`) so you can see exactly what each mark rewards. Stage 4.3 in the operations notebook also includes a **conceptual walkthrough of embedding drift** (penultimate-layer 512-dim embedding → distance-to-centroid → PSI) so you are not left
to infer it.

## Setup
```bash
pip install -r requirements.txt
# Download the casting dataset (Kaggle) and extract into data/ so you have:
#   data/.../train/{ok_front,def_front}/   and   data/.../test/{ok_front,def_front}/
#   https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
```

## Run (after implementing the TODOs)
```bash
python -m src.train          # train → MLflow → register @production
python -m src.monitoring     # statistical + embedding drift → drift_summary.json
python -m src.retrain        # drift-triggered retrain + rollback decision
pytest -q                    # API + pipeline tests
uvicorn app:app --port 8000  # serve; POST an image to /predict
```

## What you submit 
Upload these individual files; all 100 marks are graded from them:
1. `Casting_Defect_MLOps_Report` as **PDF** (Stages 1, 3, 4 + Stage-3/4 evidence).
2. `Data_Preparation.ipynb` (executed, with outputs).
3. `Model_Development_and_Tracking.ipynb` (executed, with outputs).
4. `Operations_Monitoring_and_Evidence.ipynb` (executed, with outputs).
5. `app.py`.

**Note:** Repository-generated evidence (MLflow UI, Docker build, CI green run, drift report) must be captured
**inside the notebooks/report** as screenshots, code output and summaries — we don't accept
repositories, ZIPs, HTML or JSON files.

## Acceptance targets
* `pytest` green; `/health` returns `model_loaded: true` after training.
* Test **recall on defects** is the headline metric; report F1 + ROC-AUC + confusion.
* `monitoring` flags **statistical and embedding** drift; `retrain` makes a promote/rollback decision.
* MLflow registry shows `casting_defect_classifier` with a `@production` alias.
