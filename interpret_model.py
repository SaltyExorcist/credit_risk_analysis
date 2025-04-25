import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import load_data, encode_and_scale, generate_risk_label

def generate_shap_summary(model_path: str = 'credit_risk_model.pkl',
                          data_path: str = 'german_credit_data.csv',
                          output_path: str = 'shap_summary.png'):
    model = joblib.load(model_path)
    df = load_data(data_path)
    df, _, _ = encode_and_scale(df)
    df = generate_risk_label(df)
    X = df.drop(columns=['Risk'])

    # Sample to reduce SHAP compute time
    X_sample = X.sample(n=min(100, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[1], X_sample, show=False)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"SHAP summary saved to {output_path}")

        plt.figure(figsize=(10, 8))
        shap_importance = np.abs(shap_values[1]).mean(0)
        idx_sorted = np.argsort(-shap_importance)
        plt.barh(np.array(X_sample.columns)[idx_sorted][:15], shap_importance[idx_sorted][:15])
        plt.title('Feature Importance (Mean |SHAP|)')
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_importance.png'))
        print(f"Feature importance saved to {output_path.replace('.png', '_importance.png')}")
    else:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"SHAP summary saved to {output_path}")

if __name__ == '__main__':
    generate_shap_summary()