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

    with mlflow.start_run():
        # Menggunakan parameter terbaik dari hasil tuning sebelumnya
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # Menyimpan model ke direktori lokal agar bisa di-build oleh Docker
        mlflow.sklearn.save_model(model, "model_dir")
        print("Model berhasil dilatih dan disimpan ke direktori 'model_dir'")

if __name__ == "__main__":
    train_and_save_model()
