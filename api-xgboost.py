import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def classify_with_xgboost_grid(file_path):
    """
    Klasifikasi penyakit daun teh dengan XGBoost + GridSearchCV.
    """
    try:
        print("Mencoba memuat data dari file CSV...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' tidak ditemukan.")
            
        df = pd.read_csv(file_path)
        print("Data berhasil dimuat. Ukuran dataset:", df.shape)
        
        print("\n--- DEBUGGING: Distribusi Kelas Awal ---")
        print(df['Label'].value_counts())
        
        X = df.drop(columns=['Filename', 'Label'])
        y = df['Label']
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y = y_encoded

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("\nData berhasil dibagi:")
        print(f" - Ukuran data pelatihan: {X_train.shape[0]} sampel")
        print(f" - Ukuran data pengujian: {X_test.shape[0]} sampel")
        
        print("\nMemulai GridSearch untuk XGBoost...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        xgb_model = XGBClassifier(objective='multi:softprob', eval_metric='logloss', use_label_encoder=False, random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print("\nGridSearch selesai.")
        print("Best parameters:", grid_search.best_params_)
        print("Best CV score:", grid_search.best_score_)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print("\n=== Hasil Evaluasi Model ===")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Akurasi model: {accuracy:.2%}")

        cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=best_model.classes_, yticklabels=best_model.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Prediksi Label')
        plt.ylabel('Label Sebenarnya')
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == '__main__':
    csv_file = 'extracted_features_labeled_fixed.csv' 
    classify_with_xgboost_grid(csv_file)