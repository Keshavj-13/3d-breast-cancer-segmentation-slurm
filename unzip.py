import synapseclient
import os
import gzip
import shutil

syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc1OTk0NzMzNSwiaWF0IjoxNzU5OTQ3MzM1LCJqdGkiOiIyNjg5NiIsInN1YiI6IjM1NTkyNDUifQ.qVNhlzUz76CYZ9UpERQjwSeK-XO0aSca-rXy4jrv3pLJ3hx8cu-dyUwIOtZ4rKgDgNr1TLdYBXh-VAUusBL4vqgonFDNhxKNl8-KifAXqb-fLM7BnpQUqb8VMd91ht4_WE78jUNTLDW6jk_7JVNRhFxZUJ1a0w_9GBKEIYk8TX8Wxx52yEu8NhiRsAkAo5zDHA9guo9QyjxjU3kYhusUM3Ab5awOMMndZjkYRnzmXSgJLlqYZdjA8TJugEQgk7QBkYtpPRiYqaOHThxoq1H5dYkq9EC2USJzknOCI7Dtd5t-BgL9VoP2DwLKYLZgOGK9XankSk06sIFOvpiUGnOJPA")

download_dir = "mama_mia_dataset"
os.makedirs(download_dir, exist_ok=True)

# ===== CONFIGURATION =====
FOLDERS_ALREADY_DOWNLOADED = 500  # Skip first 500 folders
MAX_FOLDERS = 1506                # Total folders to download (501-1506)
MAX_FILES_PER_FOLDER = 3          # How many files per patient folder

# Choose what to download (set to True/False)
DOWNLOAD_IMAGES = True
DOWNLOAD_SEGMENTATIONS = False    # Labels/masks
DOWNLOAD_OTHER = False            # Other folders
# =========================

downloaded_folders = 0
skipped_folders = 0

def extract_gz(file_path):
    if file_path.endswith(".gz"):
        output_path = file_path[:-3]
        with gzip.open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)
        return output_path
    return file_path

def should_skip_folder(folder_name):
    """Check if we should skip this folder based on settings"""
    folder_lower = folder_name.lower()
    
    if 'image' in folder_lower:
        return not DOWNLOAD_IMAGES
    elif 'segmentation' in folder_lower or 'label' in folder_lower or 'mask' in folder_lower:
        return not DOWNLOAD_SEGMENTATIONS
    else:
        return not DOWNLOAD_OTHER

def download_entity(entity_id, location, depth=0, parent_folder_name=""):
    global downloaded_folders, skipped_folders
    
    entity = syn.get(entity_id, downloadLocation=location, downloadFile=False)
    
    if entity.concreteType.endswith("FileEntity"):
        # Download the actual file
        entity_with_file = syn.get(entity_id, downloadLocation=location)
        file_path = entity_with_file.path
        
        if file_path.endswith(".gz"):
            file_path = extract_gz(file_path)
        print(f"Downloaded: {os.path.basename(file_path)}")
        return 1
    
    else:  # Folder
        folder_name = entity.name
        folder_path = os.path.join(location, folder_name)
        
        # Skip folder if settings say so (only at depth 1: images/, segmentations/, etc.)
        if depth == 1 and should_skip_folder(folder_name):
            print(f"Skipping folder: {folder_name} (disabled in settings)")
            return 0
        
        os.makedirs(folder_path, exist_ok=True)
        
        # Get all children
        children = list(syn.getChildren(entity_id))
        
        # Check if this is a patient folder
        is_patient_folder = depth == 2 and any(
            child['type'] == 'org.sagebionetworks.repo.model.FileEntity' 
            for child in children
        )
        
        if is_patient_folder:
            # Skip already downloaded folders
            if skipped_folders < FOLDERS_ALREADY_DOWNLOADED:
                skipped_folders += 1
                if skipped_folders % 50 == 0:  # Progress update every 50 folders
                    print(f"Skipping folder {skipped_folders}/{FOLDERS_ALREADY_DOWNLOADED}...")
                return 0
            
            # Check folder limit
            if downloaded_folders >= MAX_FOLDERS - FOLDERS_ALREADY_DOWNLOADED:
                print(f"Skipping {folder_name} (reached limit)")
                return 0
            
            downloaded_folders += 1
            print(f"\n[{parent_folder_name}] Folder {FOLDERS_ALREADY_DOWNLOADED + downloaded_folders}/{MAX_FOLDERS}: {folder_name}")
            
            # Sort and limit files
            file_children = [c for c in children if c['type'] == 'org.sagebionetworks.repo.model.FileEntity']
            file_children.sort(key=lambda x: x['name'])
            files_to_download = file_children[:MAX_FILES_PER_FOLDER]
            
            total = 0
            for i, child in enumerate(files_to_download, 1):
                print(f"  [{i}/{len(files_to_download)}] ", end="")
                total += download_entity(child['id'], folder_path, depth + 1, folder_name)
            
            skipped = len(file_children) - len(files_to_download)
            if skipped > 0:
                print(f"  Skipped {skipped} additional files")
            
            return total
        
        else:
            # Regular folder - recurse
            total = 0
            for child in children:
                if downloaded_folders >= MAX_FOLDERS - FOLDERS_ALREADY_DOWNLOADED and depth >= 2:
                    break
                total += download_entity(child['id'], folder_path, depth + 1, folder_name)
            return total

# Start downloading
print("=" * 50)
print("Download Configuration:")
print(f"  Images: {'✓' if DOWNLOAD_IMAGES else '✗'}")
print(f"  Segmentations/Labels: {'✓' if DOWNLOAD_SEGMENTATIONS else '✗'}")
print(f"  Other folders: {'✓' if DOWNLOAD_OTHER else '✗'}")
print(f"  Skipping first: {FOLDERS_ALREADY_DOWNLOADED} folders")
print(f"  Max folders: {MAX_FOLDERS}")
print(f"  Will download: {MAX_FOLDERS - FOLDERS_ALREADY_DOWNLOADED} folders (501-{MAX_FOLDERS})")
print(f"  Max files per folder: {MAX_FILES_PER_FOLDER}")
print("=" * 50)
print()

total_files = download_entity("syn60868042", download_dir)

print(f"\n{'=' * 50}")
print(f"✅ Complete!")
print(f"  Folders skipped: {skipped_folders}")
print(f"  Folders downloaded: {downloaded_folders}")
print(f"  Total files: {total_files}")
print("=" * 50)