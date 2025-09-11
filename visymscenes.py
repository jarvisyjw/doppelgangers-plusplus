import os
import zipfile

def recover_folder_from_chunks(chunks_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    zip_files = [f for f in os.listdir(chunks_folder) if f.endswith(".zip")]
    zip_files.sort()

    for zip_file in zip_files:
        zip_path = os.path.join(chunks_folder, zip_file)
        print(f"Extracting {zip_path}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(output_folder)
    
    print(f"Recovery complete. Files restored to {output_folder}")

chunks_folder = "/media/jarvis/JWDATA3/dataset/VisymScenes_chunk"
output_folder = "/media/jarvis/JWDATA3/dataset/VisymScenes"

recover_folder_from_chunks(chunks_folder, output_folder)