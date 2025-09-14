import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import os
import sys

# Menetapkan encoding stdout ke UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def segment_leaf(img):
    """
    Melakukan segmentasi untuk mengisolasi area daun.
    
    Args:
        img (np.array): Citra input dalam format BGR.
    
    Returns:
        np.array: Citra yang sudah disegmentasi dengan latar belakang hitam.
    """
    # Mengonversi BGR ke ruang warna HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Menentukan rentang warna hijau (daun) pada ruang warna HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Membuat mask biner untuk mengisolasi area hijau
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    
    # Menerapkan mask pada gambar asli
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    
    return segmented_img

def extract_features(image_path):
    """
    Ekstraksi fitur Histogram Warna dan GLCM dari sebuah gambar yang sudah disegmentasi.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"     - Gagal membaca gambar: {os.path.basename(image_path)}. Mungkin file rusak atau tidak valid.")
            return None, None
        
        # --- LANGKAH BARU: Segmentasi gambar ---
        segmented_img = segment_leaf(img)
        
        # 1. Ekstraksi Fitur Histogram Warna
        hist_features = []
        channels = cv2.split(segmented_img)
        for i, color in enumerate(('b', 'g', 'r')):
            hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist_features.extend(hist[::16])
        
        # 2. Ekstraksi Fitur GLCM
        gray_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        gray_img = np.array(gray_img / 32, dtype=np.uint8)
        
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Mengatasi kasus di mana gambar tidak memiliki fitur tekstur (misalnya, semua piksel hitam)
        if gray_img.sum() == 0:
            glcm_features = [0.0] * 16
        else:
            glcm = graycomatrix(gray_img, distances=distances, angles=angles, levels=8, symmetric=True, normed=True)
            properties = ['energy', 'correlation', 'homogeneity', 'contrast']
            glcm_features = []
            for prop in properties:
                glcm_features.extend(graycoprops(glcm, prop).flatten())
        
        return hist_features, glcm_features
    
    except Exception as e:
        print(f"     - Terjadi kesalahan saat memproses gambar: {os.path.basename(image_path)}, Error: {e}")
        return None, None

def main(base_data_folder, output_csv):
    print(f"Memulai proses ekstraksi fitur dari folder '{base_data_folder}'...")
    all_features = []
    
    column_names = ['Filename', 'Label']
    for i in range(48):
        column_names.append(f'Hist_{i+1}')
    properties = ['energy', 'correlation', 'homogeneity', 'contrast']
    angles_str = ['0deg', '45deg', '90deg', '135deg']
    for prop in properties:
        for angle_str in angles_str:
            column_names.append(f'GLCM_{prop}_{angle_str}')
    
    print(f"\nJumlah total kolom yang dibuat: {len(column_names)}")
    
    for class_folder in os.listdir(base_data_folder):
        class_label = class_folder
        class_path = os.path.join(base_data_folder, class_folder)
        
        if os.path.isdir(class_path):
            print(f"\n> Memproses kelas: '{class_label}'")
            
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_path, filename)
                    hist_feats, glcm_feats = extract_features(image_path)
                    
                    if hist_feats and glcm_feats:
                        features = [filename, class_label] + hist_feats + glcm_feats
                        all_features.append(features)

    if not all_features:
        print("\nTidak ada fitur yang berhasil diekstraksi. Pastikan folder berisi subfolder dengan gambar yang valid.")
        return

    print("\nMembuat DataFrame dan menyimpannya ke file CSV...")
    df = pd.DataFrame(all_features, columns=column_names)
    df.to_csv(output_csv, index=False)
    print(f"Proses selesai. Fitur berhasil diekstraksi dan disimpan di '{output_csv}'.")

if __name__ == "__main__":
    base_data_folder = '../Dataset diseas daun teh/Tea_Leaf_Disease/'
    output_csv_file = 'extracted_features_labeled_segmented.csv'
    
    if not os.path.exists(base_data_folder):
        print(f"Error: Folder '{base_data_folder}' tidak ditemukan.")
    else:
        main(base_data_folder, output_csv_file)