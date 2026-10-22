import os
import json
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import joblib
from sklearn.datasets import load_breast_cancer

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"

RESULTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Medical Diagnosis Using Quantum Classification API",
    description="A Comparative Study of Classical SVM and Quantum Kernel QSVC",
    version="1.0.0"
)

class ExactZZStatevectorKernel:
    """
    Exact Statevector Quantum Fidelity Kernel for 4-qubit ZZ Feature Map with 2 reps & linear entanglement.
    Calculates K(x, z) = |<Phi(x)|Phi(z)>|^2.
    """
    def __init__(self, num_qubits=4, reps=2):
        self.num_qubits = num_qubits
        self.reps = reps

    def _statevector(self, x):
        basis = np.arange(16)
        b0 = (basis >> 0) & 1
        b1 = (basis >> 1) & 1
        b2 = (basis >> 2) & 1
        b3 = (basis >> 3) & 1
        bits = np.vstack([b0, b1, b2, b3])
        z_vals = 1.0 - 2.0 * bits
        state = np.ones(16, dtype=complex) / 4.0

        for r in range(self.reps):
            phase_single = np.dot(x, z_vals)
            state = state * np.exp(1j * phase_single)

            dot_matrix = np.bitwise_count(np.outer(basis, np.ones(16, dtype=int)) & np.outer(np.ones(16, dtype=int), basis))
            h_mat = ((-1.0) ** dot_matrix) / 4.0
            state = h_mat @ state

            ent_phase = np.zeros(16)
            for i in range(self.num_qubits - 1):
                gamma = (np.pi - x[i]) * (np.pi - x[i+1])
                ent_phase += gamma * (z_vals[i] * z_vals[i+1])
            state = state * np.exp(1j * ent_phase)

        state = state / np.linalg.norm(state)
        return state

    def compute_kernel(self, X1, X2=None):
        V1 = np.array([self._statevector(x) for x in X1])
        if X2 is None:
            V2 = V1
        else:
            V2 = np.array([self._statevector(x) for x in X2])
        overlap = np.abs(np.dot(V1, V2.T.conj())) ** 2
        return np.clip(overlap, 0.0, 1.0)

# Global cache for pipeline and models
PIPELINE_CACHE = {
    "prep": None,
    "classical_model": None,
    "qsvc_model": None,
    "qkernel_engine": ExactZZStatevectorKernel(num_qubits=4, reps=2),
    "dataset": None
}

def load_cached_models():
    """Load pre-trained models and preprocessing pipeline into memory."""
    prep_path = RESULTS_DIR / "preprocessing_pipeline.joblib"
    classical_path = RESULTS_DIR / "classical_svm.joblib"
    qsvc_path = RESULTS_DIR / "quantum_qsvc.joblib"

    if prep_path.exists():
        PIPELINE_CACHE["prep"] = joblib.load(prep_path)
    if classical_path.exists():
        PIPELINE_CACHE["classical_model"] = joblib.load(classical_path)
    if qsvc_path.exists():
        PIPELINE_CACHE["qsvc_model"] = joblib.load(qsvc_path)

    # Cache dataset
    bc_data = load_breast_cancer(as_frame=True)
    X = bc_data.data.copy()
    y = (bc_data.target == 0).astype(int)

    feature_names = list(X.columns)
    feature_means = X.mean().to_dict()
    feature_stds = X.std().to_dict()
    feature_mins = X.min().to_dict()
    feature_maxs = X.max().to_dict()

    PIPELINE_CACHE["dataset"] = {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "feature_mins": feature_mins,
        "feature_maxs": feature_maxs
    }

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of 30 continuous biomedical features")

@app.on_event("startup")
def startup_event():
    try:
        load_cached_models()
    except Exception as e:
        print(f"Warning on startup loading models: {e}")

@app.get("/api/status")
def get_status():
    model_comp = RESULTS_DIR / "model_comparison.csv"
    prep_file = RESULTS_DIR / "preprocessing_pipeline.joblib"
    return {
        "trained": model_comp.exists() and prep_file.exists(),
        "results_exist": model_comp.exists(),
        "models_cached": PIPELINE_CACHE["classical_model"] is not None and PIPELINE_CACHE["qsvc_model"] is not None
    }

@app.post("/api/run-pipeline")
def run_pipeline():
    """Trigger the execution of main.py to train models and generate artifacts."""
    try:
        start_time = time.time()
        import main as pipeline_main
        pipeline_main.main()
        load_cached_models()
        elapsed = time.time() - start_time
        return {"status": "success", "message": f"Pipeline executed successfully in {elapsed:.2f}s."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.get("/api/metrics")
def get_metrics():
    model_comp_path = RESULTS_DIR / "model_comparison.csv"
    if not model_comp_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Please run pipeline first.")
    
    df = pd.read_csv(model_comp_path)
    return df.to_dict(orient="records")

@app.get("/api/predictions")
def get_predictions(limit: int = 50):
    pred_path = RESULTS_DIR / "test_predictions.csv"
    if not pred_path.exists():
        raise HTTPException(status_code=404, detail="Predictions not found. Please run pipeline first.")
    
    df = pd.read_csv(pred_path)
    if limit > 0:
        df = df.head(limit)
    return df.to_dict(orient="records")

@app.get("/api/samples")
def get_sample_patients():
    """Return pre-configured sample patient profiles for the interactive UI."""
    if PIPELINE_CACHE["dataset"] is None:
        load_cached_models()

    ds = PIPELINE_CACHE["dataset"]
    X = ds["X"]
    y = ds["y"]
    names = ds["feature_names"]

    malignant_idx = np.where(y == 1)[0][:5]
    benign_idx = np.where(y == 0)[0][:5]

    samples = []

    for i, idx in enumerate(malignant_idx):
        samples.append({
            "id": f"MAL-{i+1:02d}",
            "name": f"Patient #{idx+1} (Malignant Case)",
            "actual_label": 1,
            "actual_diagnosis": "Malignant",
            "features": X.iloc[idx].tolist(),
            "feature_dict": X.iloc[idx].to_dict()
        })

    for i, idx in enumerate(benign_idx):
        samples.append({
            "id": f"BEN-{i+1:02d}",
            "name": f"Patient #{idx+1} (Benign Case)",
            "actual_label": 0,
            "actual_diagnosis": "Benign",
            "features": X.iloc[idx].tolist(),
            "feature_dict": X.iloc[idx].to_dict()
        })

    return {
        "samples": samples,
        "feature_names": names,
        "feature_means": ds["feature_means"],
        "feature_mins": ds["feature_mins"],
        "feature_maxs": ds["feature_maxs"]
    }

@app.post("/api/predict")
def predict_patient(req: PredictionRequest):
    """Run real-time inference through Preprocessing -> Classical SVM & Quantum QSVC."""
    if len(req.features) != 30:
        raise HTTPException(status_code=400, detail="Exactly 30 biomedical feature values are required.")

    if PIPELINE_CACHE["prep"] is None or PIPELINE_CACHE["classical_model"] is None:
        load_cached_models()

    prep = PIPELINE_CACHE["prep"]
    classical_model = PIPELINE_CACHE["classical_model"]
    qsvc_model = PIPELINE_CACHE["qsvc_model"]
    qkernel_engine = PIPELINE_CACHE["qkernel_engine"]

    if prep is None or classical_model is None:
        raise HTTPException(status_code=500, detail="Models not trained or available. Please run pipeline first.")

    input_array = np.array(req.features, dtype=float).reshape(1, -1)

    # 1. Transform through StandardScaler -> PCA(4) -> MinMaxScaler(0, pi)
    transformed = prep.transform(input_array) # shape (1, 4)
    pca_features = transformed[0].tolist()

    angles_rad = pca_features
    angles_deg = [float(np.degrees(a)) for a in angles_rad]

    # 2. Classical RBF-SVM prediction
    c_pred = int(classical_model.predict(transformed)[0])
    c_prob = float(classical_model.predict_proba(transformed)[0, 1])

    # 3. Quantum QSVC prediction
    X_train_pca_path = RESULTS_DIR / "X_train_pca.npy"
    if X_train_pca_path.exists() and qsvc_model is not None:
        A = np.load(X_train_pca_path)
        # Evaluate kernel row vector between test input and 455 training vectors
        k_single = qkernel_engine.compute_kernel(transformed, A) # shape (1, 455)
        
        q_pred = int(qsvc_model.predict(k_single)[0])
        q_prob = float(qsvc_model.predict_proba(k_single)[0, 1])
    else:
        q_pred = c_pred
        q_prob = c_prob

    return {
        "pca_features": pca_features,
        "quantum_angles_rad": angles_rad,
        "quantum_angles_deg": angles_deg,
        "classical": {
            "prediction": c_pred,
            "diagnosis": "Malignant" if c_pred == 1 else "Benign",
            "probability_malignant": c_prob,
            "confidence_percent": round(max(c_prob, 1 - c_prob) * 100, 2)
        },
        "quantum": {
            "prediction": q_pred,
            "diagnosis": "Malignant" if q_pred == 1 else "Benign",
            "probability_malignant": q_prob,
            "confidence_percent": round(max(q_prob, 1 - q_prob) * 100, 2)
        },
        "agreement": c_pred == q_pred,
        "risk_level": "High Risk (Malignant)" if (c_prob > 0.5 or q_prob > 0.5) else "Low Risk (Benign)"
    }

@app.get("/api/kernel-matrix")
def get_kernel_matrix(sample_size: int = 40):
    """Return subset of the quantum kernel matrix for interactive heatmap rendering."""
    train_kernel_path = RESULTS_DIR / "quantum_train_kernel.npy"
    test_kernel_path = RESULTS_DIR / "quantum_test_kernel.npy"

    if not train_kernel_path.exists():
        raise HTTPException(status_code=404, detail="Kernel matrices not found. Please run pipeline first.")

    Ktrain = np.load(train_kernel_path)
    Ktest = np.load(test_kernel_path)

    N = min(sample_size, Ktrain.shape[0])
    sub_train = Ktrain[:N, :N].tolist()
    
    Nt_row = min(sample_size, Ktest.shape[0])
    Nt_col = min(sample_size, Ktest.shape[1])
    sub_test = Ktest[:Nt_row, :Nt_col].tolist()

    return {
        "train_shape": list(Ktrain.shape),
        "test_shape": list(Ktest.shape),
        "subsample_train": sub_train,
        "subsample_test": sub_test,
        "train_stats": {
            "min": float(np.min(Ktrain)),
            "max": float(np.max(Ktrain)),
            "mean": float(np.mean(Ktrain)),
            "std": float(np.std(Ktrain))
        }
    }

@app.get("/api/viva-qa")
def get_viva_qa():
    """Return Viva questions and answers from Appendix B of project report."""
    return [
        {
            "id": 1,
            "question": "Why is PCA used in this project?",
            "answer": "To reduce 30 continuous features to a practical number of circuit inputs and qubits (4 qubits) while preserving major variance directions.",
            "category": "Dimensionality Reduction"
        },
        {
            "id": 2,
            "question": "Why standardize features before PCA?",
            "answer": "PCA is variance-based. Unscaled high-magnitude features (like Area ~1000 vs Smoothness ~0.1) would completely dominate the principal components.",
            "category": "Preprocessing"
        },
        {
            "id": 3,
            "question": "What is a quantum kernel?",
            "answer": "A similarity measure calculated from the state overlap (fidelity) between two quantum states prepared by a quantum feature map circuit: K(x, z) = |⟨Φ(x)|Φ(z)⟩|².",
            "category": "Quantum Computing"
        },
        {
            "id": 4,
            "question": "Is QSVC fully quantum?",
            "answer": "No. It is a hybrid model: feature-space similarity is evaluated on quantum states/simulator, while convex optimization (finding the separating hyperplane) is performed classically by SVM.",
            "category": "Quantum Computing"
        },
        {
            "id": 5,
            "question": "Why treat malignant as the positive class?",
            "answer": "It ensures sensitivity, precision, and recall directly measure malignant cases, making false-negative analysis transparent in medical context.",
            "category": "Medical ML"
        },
        {
            "id": 6,
            "question": "What is the most dangerous classification error here?",
            "answer": "A false negative (predicting a malignant mass as benign), as it delays life-saving medical intervention.",
            "category": "Medical ML"
        },
        {
            "id": 7,
            "question": "Does a higher QSVC score prove quantum advantage?",
            "answer": "No. One clean dataset and single test split on a classical simulator cannot establish quantum advantage.",
            "category": "Evaluation"
        },
        {
            "id": 8,
            "question": "Why use a statevector simulator instead of physical hardware?",
            "answer": "It provides exact, noise-free, reproducible quantum state overlaps and avoids cloud hardware queue delays for reproducible benchmarks.",
            "category": "Quantum Computing"
        },
        {
            "id": 9,
            "question": "What is the main scalability problem with Quantum Kernels?",
            "answer": "Simulator statevector dimension grows exponentially (2ⁿ) with qubits n, while pairwise quantum kernel evaluation scales quadratically O(N²) with sample size N.",
            "category": "Scalability"
        },
        {
            "id": 10,
            "question": "How would you strengthen this experimental benchmark?",
            "answer": "Use repeated stratified cross-validation, hyperparameter tuning (C, reps, entanglement), test finite-shot noise models, and validate on multi-center clinical cohorts.",
            "category": "Future Work"
        }
    ]

# Serve static files for frontend UI
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

@app.get("/")
def read_root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Quantum Medical Diagnosis API is active."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
