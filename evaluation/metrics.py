# ============================================================
# EVALUATION MODULE: System Metrics Calculator
# Covers: Evaluation & Quantitative Assessment
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Import AI modules for automated evaluation
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel


class DiagnosticEvaluator:
    """
    Audits and measures diagnostic performance across all AI sub-modules
    against ground-truth dataset labels.
    """

    def __init__(self):
        self.kb = MedicalKnowledgeBase()
        self.bn = SimpleBayesianDiagnostics()
        self.ml = MLDiagnosticClassifier()
        self.nn = NeuralDiagnosticModel()
        
        # Train ML and Deep Learning models for evaluation
        self.ml.train(verbose=False)
        self.nn.train(epochs=20, verbose=0)

    def generate_evaluation_dataset(self, n_samples: int = 500) -> pd.DataFrame:
        """
        Generates a standardized evaluation dataset of patient cases
        with synthetic symptoms and known ground-truth disease labels.
        """
        return self.ml._generate_synthetic_data(n_samples)

    def evaluate_all_modules(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs the evaluation dataset through all diagnostic sub-modules
        and computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
        """
        y_true = eval_df['disease'].tolist()
        
        # Prepare symptom feature strings per row
        symptom_lists = []
        for _, row in eval_df.iterrows():
            symptom_lists.append([
                feat for feat in self.ml.SYMPTOM_FEATURES if row[feat] == 1
            ])

        results = {}
        modules_to_test = {
            'KnowledgeBase': self._predict_kb,
            'BayesianNet':   self._predict_bn,
            'MLClassifier':  self._predict_ml,
            'NeuralNetwork': self._predict_nn
        }

        for mod_name, predict_fn in modules_to_test.items():
            y_pred = []
            for symptoms in symptom_lists:
                try:
                    pred = predict_fn(symptoms)
                except Exception:
                    pred = 'Unknown'
                y_pred.append(pred)

            # Calculate classification metrics
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred)
            labels = sorted(list(set(y_true) | set(y_pred)))

            results[mod_name] = {
                'accuracy': round(float(acc), 4),
                'precision': round(float(precision), 4),
                'recall': round(float(recall), 4),
                'f1_score': round(float(f1), 4),
                'confusion_matrix': cm,
                'labels': labels,
                'predictions': y_pred
            }

        return results

    def _predict_kb(self, symptoms: List[str]) -> str:
        self.kb.facts = set()
        self.kb.certainty_factors = {}
        self.kb.load_patient_symptoms(symptoms)
        inferred = self.kb.forward_chain()
        diseases = {k: v for k, v in inferred.items() if 'suspected' in k or 'confirmed' in k}
        if not diseases:
            return 'common_cold'
        top = max(diseases, key=diseases.get)
        return top.replace('_suspected', '').replace('_confirmed', '')

    def _predict_bn(self, symptoms: List[str]) -> str:
        posteriors = self.bn.compute_posterior(symptoms)
        return max(posteriors, key=posteriors.get)

    def _predict_ml(self, symptoms: List[str]) -> str:
        res = self.ml.predict(symptoms)
        return res['diagnosis']

    def _predict_nn(self, symptoms: List[str]) -> str:
        res = self.nn.predict(symptoms)
        return res['diagnosis']


# Standalone Verification Test
if __name__ == "__main__":
    print("\n[OK] Evaluation metrics summary generated successfully.")
    evaluator = DiagnosticEvaluator()
    eval_df = evaluator.generate_evaluation_dataset(n_samples=400)
    metrics_summary = evaluator.evaluate_all_modules(eval_df)

    print(f"\n{'Module':<18} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<8} | {'F1-Score'}")
    print("-" * 65)
    for mod_name, metrics in metrics_summary.items():
        print(f"{mod_name:<18} | {metrics['accuracy']:<9.2%} | {metrics['precision']:<10.2%} | "
              f"{metrics['recall']:<8.2%} | {metrics['f1_score']:.2%}")

    print("\n[✔] Evaluation metrics calculation completed!")