# ============================================================
# EVALUATION MODULE: Visualizations Generator
# Covers: System Performance Visualization
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any


class PerformanceVisualizer:
    """
    Generates comparison bar charts, confusion matrices, and ROC-AUC style 
    visual summaries for system audit reports.
    """

    @staticmethod
    def plot_module_comparison(metrics_results: Dict[str, Any], save_path: str = "module_comparison.png"):
        """Plots bar chart comparing Accuracy, Precision, Recall, and F1-Score across modules"""
        modules = list(metrics_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(modules))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            scores = [metrics_results[m][metric] * 100 for m in modules]
            ax.bar(x + (i - 1.5) * width, scores, width, label=metric.replace('_', ' ').title())

        ax.set_ylabel('Score (%)', fontweight='bold')
        ax.set_title('AI Module Performance Benchmark Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modules, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend(loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Saved chart: {save_path}")

    @staticmethod
    def plot_all_confusion_matrices(metrics_results: Dict[str, Any], save_path: str = "confusion_matrices.png"):
        """Plots confusion matrix heatmaps for each classifier sub-module"""
        n_modules = len(metrics_results)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (mod_name, data) in enumerate(metrics_results.items()):
            cm = data['confusion_matrix']
            labels = data['labels']
            ax = axes[idx]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=labels, yticklabels=labels, cbar=False)
            ax.set_title(f"{mod_name} Confusion Matrix", fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

        plt.suptitle("Sub-Module Confusion Matrix Heatmaps", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Saved heatmaps: {save_path}")


# Standalone Verification Test
if __name__ == "__main__":
    from evaluation.metrics import DiagnosticEvaluator

    print("--- Running Evaluation Visualizations Test ---")
    evaluator = DiagnosticEvaluator()
    eval_df = evaluator.generate_evaluation_dataset(n_samples=300)
    metrics_summary = evaluator.evaluate_all_modules(eval_df)

    visualizer = PerformanceVisualizer()
    visualizer.plot_module_comparison(metrics_summary)
    visualizer.plot_all_confusion_matrices(metrics_summary)

    print("\n[✔] Evaluation visualization graphics successfully generated!")