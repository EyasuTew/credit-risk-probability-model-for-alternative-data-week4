# Credit Risk Model for Bati Bank Buy-Now-Pay-Later Service

## Overview
[Previous content...]

## Credit Scoring Business Understanding
[Your previous section here]

## Installation & Setup
1. Clone repo: `git clone <repo>`
2. Install deps: `pip install -r requirements.txt`
3. Download data to `data/raw/data.csv`.
4. Process data: `python src/data_processing.py`
5. Train model: `python src/train.py`
6. Run API: `uvicorn src.api.main:app --reload`
7. Test: `python -m pytest tests/ -v`

## Model Details
- Proxy: RFM-based binary label (bad/good).
- Risk Model: Logistic Regression (PD).
- Score: Scaled from prob (low prob = high score).
- Loan Optimizer: Rules based on score.

## Deployment
Dockerfile and docker-compose.yml provided for CI/CD.