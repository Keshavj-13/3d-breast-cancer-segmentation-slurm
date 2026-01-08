import synapseclient
import os
import gzip
import shutil

syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc1OTk0NzMzNSwiaWF0IjoxNzU5OTQ3MzM1LCJqdGkiOiIyNjg5NiIsInN1YiI6IjM1NTkyNDUifQ.qVNhlzUz76CYZ9UpERQjwSeK-XO0aSca-rXy4jrv3pLJ3hx8cu-dyUwIOtZ4rKgDgNr1TLdYBXh-VAUusBL4vqgonFDNhxKNl8-KifAXqb-fLM7BnpQUqb8VMd91ht4_WE78jUNTLDW6jk_7JVNRhFxZUJ1a0w_9GBKEIYk8TX8Wxx52yEu8NhiRsAkAo5zDHA9guo9QyjxjU3kYhusUM3Ab5awOMMndZjkYRnzmXSgJLlqYZdjA8TJugEQgk7QBkYtpPRiYqaOHThxoq1H5dYkq9EC2USJzknOCI7Dtd5t-BgL9VoP2DwLKYLZgOGK9XankSk06sIFOvpiUGnOJPA")

download_dir = "mama_mia_dataset"
os.makedirs(download_dir, exist_ok=True)

def extract_gz(file_path):
    if file_path.endswith(".gz"):
        output_path = file_path[:-3]
        with gzip.open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)
        return output_path
    return file_path

def download_entity(entity_id, location):
    entity = syn.get(entity_id, downloadLocation=location)

    if entity.concreteType.endswith("FileEntity"):
        file_path = entity.path
        if file_path.endswith(".gz"):
            file_path = extract_gz(file_path)
        print(f"Downloaded: {file_path} ({os.path.getsize(file_path)} bytes)")
        return 1
    
    folder_name = entity.name
    folder_path = os.path.join(location, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    total = 0
    children = syn.getChildren(entity_id)

    for child in children:
        child_name = child["name"]

        # Skip any folder whose name contains the word images
        if child["type"] == "org.sagebionetworks.repo.model.Folder" and "image" in child_name.lower():
            print(f"Skipping folder: {child_name}")
            continue

        total += download_entity(child["id"], folder_path)

    return total

total_files = download_entity("syn60868042", download_dir)
print(f"Total files downloaded and extracted: {total_files}")
