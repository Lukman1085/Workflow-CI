import os
import shutil
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def train_and_save_model():
    # Membersihkan folder model lama jika ada untuk mencegah konflik
    if os.path.exists("model_dir"):
        shutil.rmtree("model_dir")

    # Memuat dataset
    train_data = pd.read_csv("ai_impact_preprocessing/train_processed.csv")
    X_train = train_data.drop(columns=['Burnout_Risk_Level'])
    y_train = train_data['Burnout_Risk_Level']

    custom_env = {
        "name": "burnout-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.12.7",  # Memaksa Docker menggunakan Python 3.12.7
            "pip",
            {"pip": ["mlflow==2.19.0", "scikit-learn", "pandas", "numpy"]},
        ],
    }

    with mlflow.start_run():
        # Menggunakan parameter terbaik dari hasil tuning sebelumnya
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        mlflow.sklearn.save_model(
            sk_model=model, 
            path="model_dir", 
            conda_env=custom_env
        )
        print("✅ Model CI berhasil dilatih dan disimpan dengan env Python 3.12.7")

if __name__ == "__main__":
    train_and_save_model()
