from pathlib import Path
import json, time, warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
 confusion_matrix, f1_score, precision_score, recall_score,
 roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

# Qiskit & Custom Quantum Statevector Engine
USE_QISKIT = True
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
except Exception as e:
    print(f"Note: Using exact 4-qubit statevector fidelity kernel engine ({e})")
    USE_QISKIT = False

RANDOM_STATE, TEST_SIZE, N_COMPONENTS = 42, 0.20, 4
OUT = Path("results"); OUT.mkdir(exist_ok=True)

class ExactZZStatevectorKernel:
    """
    Exact Statevector Quantum Fidelity Kernel for 4-qubit ZZ Feature Map with 2 reps & linear entanglement.
    Calculates K(x, z) = |<Phi(x)|Phi(z)>|^2.
    """
    def __init__(self, num_qubits=4, reps=2):
        self.num_qubits = num_qubits
        self.reps = reps
        self.dim = 2 ** num_qubits

    def _statevector(self, x):
        """Construct statevector in C^16 for 4D vector x in [0, pi]^4."""
        # Basis states 0..15
        basis = np.arange(16)
        # Bit representations: b_0, b_1, b_2, b_3
        b0 = (basis >> 0) & 1
        b1 = (basis >> 1) & 1
        b2 = (basis >> 2) & 1
        b3 = (basis >> 3) & 1
        bits = np.vstack([b0, b1, b2, b3]) # shape (4, 16)
        
        # Z eigenvalues: (-1)^b in {+1, -1}
        z_vals = 1.0 - 2.0 * bits # shape (4, 16)

        # Hadamard state init: 1/sqrt(16) * sum |k>
        state = np.ones(16, dtype=complex) / 4.0

        for r in range(self.reps):
            # 1. Single qubit rotations: U_Z(x_i) = exp(i * x_i * Z_i)
            # Phase for each basis state k: sum_i x_i * z_{i, k}
            phase_single = np.dot(x, z_vals)
            state = state * np.exp(1j * phase_single)

            # 2. Hadamard layer: H^tensor 4
            # H matrix element <k|H^4|m> = 1/4 * (-1)^(k dot m)
            dot_matrix = np.bitwise_count(np.outer(basis, np.ones(16, dtype=int)) & np.outer(np.ones(16, dtype=int), basis))
            h_mat = ((-1.0) ** dot_matrix) / 4.0
            state = h_mat @ state

            # 3. Entangling rotations: U_ZZ(x_i, x_{i+1}) = exp(i * (pi - x_i)(pi - x_{i+1}) * Z_i Z_{i+1})
            ent_phase = np.zeros(16)
            for i in range(self.num_qubits - 1):
                gamma = (np.pi - x[i]) * (np.pi - x[i+1])
                ent_phase += gamma * (z_vals[i] * z_vals[i+1])
            state = state * np.exp(1j * ent_phase)

        # Normalize statevector
        state = state / np.linalg.norm(state)
        return state

    def compute_kernel(self, X1, X2=None):
        """Compute Gram matrix K between arrays X1 and X2."""
        V1 = np.array([self._statevector(x) for x in X1]) # shape (N1, 16)
        if X2 is None:
            V2 = V1
        else:
            V2 = np.array([self._statevector(x) for x in X2]) # shape (N2, 16)
        
        # Overlap matrix: O_ij = <V1_i | V2_j>
        overlap = np.abs(np.dot(V1, V2.T.conj())) ** 2
        return np.clip(overlap, 0.0, 1.0)

def specificity(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    return tn / (tn + fp) if tn + fp else 0.0

def evaluate(name, y, p, score):
    return {
        "model": name,
        "accuracy": accuracy_score(y, p),
        "precision_malignant": precision_score(y, p, zero_division=0),
        "sensitivity_recall": recall_score(y, p, zero_division=0),
        "specificity": specificity(y, p),
        "f1_malignant": f1_score(y, p, zero_division=0),
        "roc_auc": roc_auc_score(y, score)
    }

def cm_plot(y, p, title, name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y, p), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=300)
    plt.close()

def metric_comparison_plot(cmet, qmet):
    metrics = ["accuracy", "precision_malignant", "sensitivity_recall", "specificity", "f1_malignant", "roc_auc"]
    labels = ["Accuracy", "Precision (M)", "Sensitivity (M)", "Specificity (B)", "F1 Score (M)", "ROC-AUC"]
    c_vals = [cmet[m] for m in metrics]
    q_vals = [qmet[m] for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, c_vals, width, label='Classical RBF-SVM', color='#3b82f6')
    ax.bar(x + width/2, q_vals, width, label='Quantum QSVC', color='#8b5cf6')
    
    ax.set_ylabel('Score')
    ax.set_title('Classical RBF-SVM vs Quantum QSVC Metric Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i in range(len(labels)):
        ax.annotate(f"{c_vals[i]:.3f}", (x[i] - width/2, c_vals[i] + 0.01), ha='center', fontsize=8)
        ax.annotate(f"{q_vals[i]:.3f}", (x[i] + width/2, q_vals[i] + 0.01), ha='center', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(OUT / "model_metric_comparison.png", dpi=300)
    plt.close()

def generate_circuit_text():
    """Generate textual representation of 4-qubit ZZ Feature Map circuit."""
    return """q_0: ───[ H ]───[ Rz(2*x[0]) ]───■───────────────────■───────────────────────────────────────────────────[ H ]───[ Rz(2*x[0]) ]───■───────────────────■─────────────────────────────────
                                 │                   │                                                                   │                   │                                 
q_1: ───[ H ]───[ Rz(2*x[1]) ]───■───[ Rz(2*g01) ]───■───■───────────────────■───────────────────────────[ H ]───[ Rz(2*x[1]) ]───■───[ Rz(2*g01) ]───■───■───────────────────■─────────────────
                                                         │                   │                                                                   │                   │                 
q_2: ───[ H ]───[ Rz(2*x[2]) ]───────────────────────────■───[ Rz(2*g12) ]───■───■───────────────────■───[ H ]───[ Rz(2*x[2]) ]───────────────────────────■───[ Rz(2*g12) ]───■───■─────────────
                                                                                 │                   │                                                                       │             
q_3: ───[ H ]───[ Rz(2*x[3]) ]───────────────────────────────────────────────────■───[ Rz(2*g23) ]───■───[ H ]───[ Rz(2*x[3]) ]───────────────────────────────────────────────────■───[ Rz(2*g23
                                                                                                                                                                                           
Where: g_ij = (pi - x[i]) * (pi - x[j])
Entanglement: Linear (q0-q1, q1-q2, q2-q3) | Repetitions: 2 | Feature dimension: 4
"""

def main():
    warnings.filterwarnings("ignore")
    print("[1/6] Loading Wisconsin Diagnostic Breast Cancer dataset...")
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = (data.target == 0).astype(int) # malignant becomes 1

    print(f"Total samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Class distribution: Benign (0): {sum(y==0)}, Malignant (1): {sum(y==1)}")

    print("[2/6] Stratified 80/20 train/test split & preprocessing pipeline...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    prep = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)),
        ("angle", MinMaxScaler(feature_range=(0, np.pi)))
    ])

    A = prep.fit_transform(X_train)
    B = prep.transform(X_test)

    # Save preprocessing pipeline and fitted objects
    joblib.dump(prep, OUT / "preprocessing_pipeline.joblib")
    np.save(OUT / "X_train_pca.npy", A)
    np.save(OUT / "X_test_pca.npy", B)
    np.save(OUT / "y_train.npy", y_train.values)
    np.save(OUT / "y_test.npy", y_test.values)

    print("[3/6] Training Classical RBF-SVM Baseline...")
    classical = SVC(
        kernel="rbf", C=2.0, probability=True,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    t = time.perf_counter()
    classical.fit(A, y_train)
    ct = time.perf_counter() - t

    cp = classical.predict(B)
    cs = classical.predict_proba(B)[:, 1]

    cmet = evaluate("Classical RBF SVM", y_test, cp, cs)
    cmet.update({"training_time_seconds": ct, "kernel_time_seconds": 0.0})
    cm_plot(y_test, cp, "Classical RBF-SVM", "classical_confusion_matrix.png")
    joblib.dump(classical, OUT / "classical_svm.joblib")

    print("[4/6] Computing 4-Qubit Quantum ZZ Fidelity Kernel Matrix...")
    qkernel_engine = ExactZZStatevectorKernel(num_qubits=4, reps=2)
    
    t = time.perf_counter()
    print("Evaluating training kernel matrix K(A, A) (455x455)...")
    Ktrain = qkernel_engine.compute_kernel(A)
    print("Evaluating testing kernel matrix K(B, A) (114x455)...")
    Ktest = qkernel_engine.compute_kernel(B, A)
    kt = time.perf_counter() - t

    np.save(OUT / "quantum_train_kernel.npy", Ktrain)
    np.save(OUT / "quantum_test_kernel.npy", Ktest)
    (OUT / "quantum_circuit.txt").write_text(generate_circuit_text(), encoding="utf-8")

    print("[5/6] Training Quantum Support Vector Classifier (QSVC)...")
    qsvc = SVC(
        kernel="precomputed", C=2.0, probability=True,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    t = time.perf_counter()
    qsvc.fit(Ktrain, y_train)
    qt = time.perf_counter() - t

    qp = qsvc.predict(Ktest)
    qs = qsvc.predict_proba(Ktest)[:, 1]

    qmet = evaluate("Quantum QSVC", y_test, qp, qs)
    qmet.update({"training_time_seconds": qt, "kernel_time_seconds": kt})
    cm_plot(y_test, qp, "Quantum QSVC", "quantum_confusion_matrix.png")
    joblib.dump(qsvc, OUT / "quantum_qsvc.joblib")

    print("[6/6] Generating evaluation reports and ROC curves...")
    results = pd.DataFrame([cmet, qmet])
    results.to_csv(OUT / "model_comparison.csv", index=False)

    pd.DataFrame({
        "actual": np.asarray(y_test),
        "classical_prediction": cp,
        "classical_probability": cs,
        "quantum_prediction": qp,
        "quantum_probability": qs
    }).to_csv(OUT / "test_predictions.csv", index=False)

    metric_comparison_plot(cmet, qmet)

    fpr1, tpr1, _ = roc_curve(y_test, cs)
    fpr2, tpr2, _ = roc_curve(y_test, qs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr1, tpr1, label=f"RBF AUC={cmet['roc_auc']:.3f}", color='#3b82f6', lw=2)
    plt.plot(fpr2, tpr2, label=f"QSVC AUC={qmet['roc_auc']:.3f}", color='#8b5cf6', lw=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("ROC Comparison: Classical RBF-SVM vs Quantum QSVC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve_comparison.png", dpi=300)
    plt.close()

    print("\n" + "="*60)
    print("RESULTS SUMMARY:")
    print("="*60)
    print(results.round(4).to_string(index=False))
    print("="*60)
    print("All artifacts successfully saved to results/ folder.")

if __name__ == "__main__":
    main()
