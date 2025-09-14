import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

def classify_with_svm(file_path):
    """
    Melakukan klasifikasi penyakit daun teh dengan Support Vector Machine (SVM).
    
    Args:
        file_path (str): Path ke file CSV yang berisi fitur dan label.
    """
    try:
        # 1. Memuat data dari CSV
        print("Mencoba memuat data dari file CSV...")
        df = pd.read_csv(file_path)
        print("Data berhasil dimuat. Ukuran dataset:", df.shape)

        # --- FUNGSI DEBUGGING TAMBAHAN: Memeriksa distribusi kelas ---
        print("\n--- DEBUGGING: Distribusi Kelas Awal ---")
        print(df['Label'].value_counts())
        
        # 2. Memisahkan fitur (X) dan label (y)
        X = df.drop(columns=['Filename', 'Label'])
        # Encode label menjadi angka
        le = LabelEncoder()
        y = le.fit_transform(df['Label'])
        print("\n--- DEBUGGING: Mapping Label ---")
        print(dict(zip(le.classes_, le.transform(le.classes_))))

        # --- FUNGSI DEBUGGING TAMBAHAN: Memeriksa jumlah fitur dan sampel ---
        print("\n--- DEBUGGING: Informasi Data ---")
        print(f"Jumlah sampel: {X.shape[0]}")
        print(f"Jumlah fitur: {X.shape[1]}")
        print("Nama-nama fitur:", list(X.columns))

        # 3. Membagi data menjadi data pelatihan dan pengujian (rasio 80:20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("\nData berhasil dibagi:")
        print(f" - Ukuran data pelatihan: {X_train.shape[0]} sampel")
        print(f" - Ukuran data pengujian: {X_test.shape[0]} sampel")
        
        # --- FUNGSI DEBUGGING TAMBAHAN: Memeriksa distribusi kelas setelah split ---
        print("\n--- DEBUGGING: Distribusi Kelas Setelah Split ---")
        print("\n--- DEBUGGING: Distribusi Kelas Setelah Split ---")
        print("Distribusi kelas pada data pelatihan:")
        print(pd.Series(y_train).value_counts())
        print("\nDistribusi kelas pada data pengujian:")
        print(pd.Series(y_test).value_counts())
        # 4. Inisialisasi dan melatih model SVM
        print("\nMemulai pelatihan model SVM...")
        param_grid = {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'shrinking': [True, False]
        }

        grid_search = GridSearchCV(
            SVC(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        svm_model = grid_search.best_estimator_
        print("Best parameters from GridSearchCV:", grid_search.best_params_)
        svm_model.fit(X_train, y_train)
        print("Pelatihan model SVM selesai.")
        
        # --- FUNGSI DEBUGGING TAMBAHAN: Memeriksa fitur terpenting (untuk model linear) ---
        print("\n--- DEBUGGING: Koefisien Fitur SVM (Jika kernel linear) ---")
        if svm_model.kernel == 'linear' and X.shape[1] > 0:
            coef = svm_model.coef_[0]
            feature_names = X.columns
            coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
            coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
            top_features = coef_df.sort_values(by='abs_coefficient', ascending=False).head(10)
            print("10 fitur terpenting:")
            print(top_features)

        # 5. Membuat prediksi pada data pengujian
        y_pred = svm_model.predict(X_test)

        # 6. Evaluasi model dengan Confusion Matrix dan Akurasi
        print("\n=== Hasil Evaluasi Model ===")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Akurasi model: {accuracy:.2%}")

        # Menampilkan Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Prediksi Label')
        plt.ylabel('Label Sebenarnya')
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{file_path}' tidak ditemukan. Pastikan nama file dan path sudah benar.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Penggunaan fungsi
if __name__ == '__main__':
    csv_file = 'extracted_features_labeled_fixed.csv' 
    classify_with_svm(csv_file)