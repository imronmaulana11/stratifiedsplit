import os
import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Path ke folder dataset
base_folder = r'D:/KULIAH/SEMESTER 7/SKRIPSI/DATASET/New dataset/DATASET BARU'
images_folder = os.path.join(base_folder, 'images')
labels_folder = os.path.join(base_folder, 'labels')

# Output folder
output_folder = os.path.join(base_folder, 'split')

# Fungsi untuk mendefinisikan label dari file dalam folder label
def get_label_from_file(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        # Ekstrak label pertama dari setiap line
        labels = [int(line.strip().split()[0]) for line in lines if line.strip()]
        # Jika label 1 (accident) tersedia, anggap itu accident
        if 1 in labels:
            return 1
        # Jika label 2 (weapon) tersedia, anggap itu weapon
        elif 2 in labels:
            return 2
        # Jika hanya label 0 yang tersedia, bukan accident
        else:
            return 0

# Fungsi untuk mengambil semua file dan labelnya dengan improved matching
def collect_files_and_labels():
    image_files = []
    labels = []
    
    # Temukan semua file gambar dengan format jpg dan png
    jpg_files = glob.glob(os.path.join(images_folder, '*.jpg'))
    png_files = glob.glob(os.path.join(images_folder, '*.png'))
    
    # Gabungkan semua file gambar
    all_image_files = jpg_files + png_files
    
    print(f"Found {len(jpg_files)} JPG files and {len(png_files)} PNG files")
    
    # Dapatkan semua file label dari folder label
    label_files = glob.glob(os.path.join(labels_folder, '*.txt'))
    label_basenames = [os.path.basename(f) for f in label_files]
    label_dict = {os.path.splitext(name)[0]: os.path.join(labels_folder, name) for name in label_basenames}
    
    print(f"Found {len(label_files)} label files")
    
    missing_files = 0
    found_files = 0
    
    for img_file in all_image_files:
        # Dapatkan nama file tanpa ekstensi
        img_basename = os.path.basename(img_file)
        img_name_without_ext = os.path.splitext(img_basename)[0]
        
        # Coba kemungkinan nama file label yang berbeda
        possible_keys = [
            img_name_without_ext,                  # Format standar
            os.path.basename(img_file),            # Nama file Full dengan ekstensi
            img_name_without_ext.split('.rf.')[0]  # Nama sebelum .rf.
        ]
        
       # Untuk file augmentasi yang mengandung 'rf', coba temukan pola file aslinya
        if '.rf.' in img_name_without_ext:
            base_name = img_name_without_ext.split('.rf.')[0]
            possible_keys.append(base_name)
        
        found_label = False
        for key in possible_keys:
            if key in label_dict:
                try:
                    label = get_label_from_file(label_dict[key])
                    image_files.append(img_file)
                    labels.append(label)
                    found_label = True
                    found_files += 1
                    break
                except Exception as e:
                    print(f"Error reading file {label_dict[key]}: {e}")
        
        if not found_label:
            # Coba untuk temukan partial match (sangat berguna untuk gambar yang telah diaugmentasi)
            found_partial = False
            for label_key in label_dict.keys():
                if img_name_without_ext.startswith(label_key) or label_key.startswith(img_name_without_ext.split('-')[0]):
                    try:
                        label = get_label_from_file(label_dict[label_key])
                        image_files.append(img_file)
                        labels.append(label)
                        found_partial = True
                        found_files += 1
                        print(f"Found partial match: {label_key} for {img_name_without_ext}")
                        break
                    except Exception as e:
                        print(f"Error reading file {label_dict[label_key]}: {e}")
            
            if not found_partial:
                missing_files += 1
                print(f"Label file not found for {img_file}")
    
    print(f"Total images with matching labels: {found_files}")
    print(f"Total images without labels: {missing_files}")
    
    return image_files, labels

# Fungsi untuk membuat folder output jika tidak tersedia
def create_output_folders():
    for split in ['train', 'val', 'test']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(output_folder, split, folder), exist_ok=True)

# Fungsi untuk memindahkan file ke folder
def move_files(image_files, labels, split_indices, split_name):
    count = 0
    for idx in split_indices:
        img_file = image_files[idx]
        img_name = os.path.basename(img_file)
        img_name_without_ext = os.path.splitext(img_name)[0]
        
        # Dapatkan label file path
        # Temukan matching label file dalam folder label
        matching_label_file = None
        for label_file in glob.glob(os.path.join(labels_folder, '*.txt')):
            label_basename = os.path.splitext(os.path.basename(label_file))[0]
            if label_basename == img_name_without_ext or img_name_without_ext.startswith(label_basename) or \
                (('.rf.' in img_name_without_ext) and label_basename == img_name_without_ext.split('.rf.')[0]):
                matching_label_file = label_file
                break
        
        if matching_label_file is None:
            print(f"Warning: No matching label file found for {img_name} during move operation")
            continue
        
        # Path destinasi
        dest_img_path = os.path.join(output_folder, split_name, 'images', img_name)
        dest_label_path = os.path.join(output_folder, split_name, 'labels', os.path.basename(matching_label_file))
        
        # Copy file gambar and label
        shutil.copy2(img_file, dest_img_path)
        shutil.copy2(matching_label_file, dest_label_path)
        count += 1
    
    print(f"Copied {count} pairs of files to {split_name} folder")

# Fungsi utama
def main():
    print(f"Looking for images in: {images_folder}")
    print(f"Looking for labels in: {labels_folder}")
    print("Collecting data and labels...")
    
    # Cek jika folder tersedia
    if not os.path.exists(images_folder):
        print(f"Images folder not found: {images_folder}")
        return
    if not os.path.exists(labels_folder):
        print(f"Labels folder not found: {labels_folder}")
        return
    
    image_files, labels = collect_files_and_labels()
    
    if not image_files:
        print("No files found. Check your folder structure.")
        return
    
    # Convert ke numpy array
    labels = np.array(labels)
    
    print(f"Total dataset: {len(image_files)} images")
    print(f"Initial distribution: Class 0 (Not Accident): {np.sum(labels == 0)}, " 
          f"Class 1 (Accident): {np.sum(labels == 1)}, "
          f"Class 2 (Weapon): {np.sum(labels == 2)}")
    
    # Buat folder output
    print("Creating output folder structure...")
    create_output_folders()
    
    # Split dataset: train (85%), temp (15%)
    print("Splitting dataset with stratification...")
    train_indices, temp_indices = train_test_split(
        range(len(image_files)), 
        test_size=0.15, 
        random_state=42, 
        stratify=labels
    )
    
    # From temp (15%), split into validation (10% of total) and test (5% of total)
    temp_labels = [labels[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.33,  # 5%/(10%+5%) = 1/3 = 0.33
        random_state=42,
        stratify=temp_labels
    )
    
    # Move files to appropriate folders
    print("Copying files to train folder...")
    move_files(image_files, labels, train_indices, 'train')
    print("Copying files to validation folder...")
    move_files(image_files, labels, val_indices, 'val')
    print("Copying files to test folder...")
    move_files(image_files, labels, test_indices, 'test')
    
    # Print distribution statistics
    print("\nDistribution statistics:")
    print(f"Total dataset: {len(image_files)} images")
    print(f"Train set: {len(train_indices)} images ({len(train_indices)/len(image_files)*100:.2f}%)")
    print(f"Validation set: {len(val_indices)} images ({len(val_indices)/len(image_files)*100:.2f}%)")
    print(f"Test set: {len(test_indices)} images ({len(test_indices)/len(image_files)*100:.2f}%)")
    
    # Print class distribution
    train_class_dist = [labels[i] for i in train_indices]
    val_class_dist = [labels[i] for i in val_indices]
    test_class_dist = [labels[i] for i in test_indices]
    
    # Convert to DataFrame for better display
    dist_df = pd.DataFrame({
        'Class': ['Not Accident (0)', 'Accident (1)', 'Weapon (2)'],
        'Train': [train_class_dist.count(0), train_class_dist.count(1), train_class_dist.count(2)],
        'Validation': [val_class_dist.count(0), val_class_dist.count(1), val_class_dist.count(2)],
        'Test': [test_class_dist.count(0), test_class_dist.count(1), test_class_dist.count(2)],
        'Total': [labels.tolist().count(0), labels.tolist().count(1), labels.tolist().count(2)]
    })
    
    print("\nClass distribution:")
    print(dist_df)
    
    print("\nDataset split process completed!")
    print(f"Data has been split and saved in folder: {output_folder}")

if __name__ == "__main__":
    main()