from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix

# Memuat data gambar langsung dari folder
def load_image_data(data_dir):
    """
    Memuat dataset gambar dari direktori yang terstruktur.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Error: Folder '{data_dir}' tidak ditemukan.")

    # Menggunakan tf.keras.utils.image_dataset_from_directory untuk kemudahan
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(180, 180),
        batch_size=32
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(180, 180),
        batch_size=32
    )
    
    class_names = train_ds.class_names
    print(f"\nKelas yang ditemukan: {class_names}")
    
    return train_ds, val_ds, class_names

def build_and_train_cnn(train_ds, val_ds, num_classes):
    """
    Membangun dan melatih model CNN.
    """
    # Membuat model CNN sederhana
    model = keras.Sequential([
        # Lapisan Konvolusi dan Pooling pertama
        layers.Rescaling(1./255, input_shape=(180, 180, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Lapisan Konvolusi dan Pooling kedua
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Lapisan Konvolusi dan Pooling ketiga
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Meratakan output untuk lapisan fully connected
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # Mengompilasi model
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Menampilkan ringkasan model
    model.summary()

    # Melatih model
    epochs = 5  # Ubah jumlah epochs untuk pelatihan lebih lama
    print(f"\nMemulai pelatihan untuk {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    return model, history

def evaluate_model(model, val_ds, class_names):
    """
    Mengevaluasi model dan menampilkan confusion matrix dengan perbaikan.
    """
    print("\n--- Hasil Evaluasi Model ---")

    # --- Perbaikan: Mendapatkan label aktual dan prediksi dalam urutan yang benar ---
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    # Menghitung akurasi
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Akurasi model: {accuracy:.2%}")
    
    # Menampilkan Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Prediksi Label')
    plt.ylabel('Label Sebenarnya')
    plt.show()

if __name__ == '__main__':
    data_directory = 'D:/local Disk D/UGM/Semester 2/Sistem Cerdas/Tugas UTs/Dataset diseas daun teh/Tea_Leaf_Disease/'
    
    # Memuat data
    train_ds, val_ds, class_names = load_image_data(data_directory)
    num_classes = len(class_names)
    
    # Membangun dan melatih model
    model, history = build_and_train_cnn(train_ds, val_ds, num_classes)
    
    # Mengevaluasi model
    evaluate_model(model, val_ds, class_names)