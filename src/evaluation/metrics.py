import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

def compute_classification_metrics(y_true, y_probs, threshold=0.5):
    """
    Computes comprehensive clinical evaluation metrics for binary classification:
    - Accuracy
    - Precision (Positive Predictive Value)
    - Recall / Sensitivity (True Positive Rate)
    - Specificity (True Negative Rate)
    - F1-Score (Harmonic mean of precision & recall)
    - ROC-AUC (Area Under Receiver Operating Characteristic Curve)
    - PR-AUC / Average Precision (Area Under Precision-Recall Curve)
    - Confusion Matrix [TN, FP, FN, TP]
    """
    y_true = np.array(y_true).astype(int)
    y_probs = np.array(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)
    
    # Base metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Threshold-independent AUC metrics
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except:
        pr_auc = 0.0
        
    # Confusion matrix & Specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        spec = 0.0
        
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'sensitivity_recall': float(rec),
        'specificity': float(spec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'threshold': float(threshold),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    }
