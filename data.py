import synapseclient
import os
from tqdm import tqdm
# Login
syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc1OTk0NzMzNSwiaWF0IjoxNzU5OTQ3MzM1LCJqdGkiOiIyNjg5NiIsInN1YiI6IjM1NTkyNDUifQ.qVNhlzUz76CYZ9UpERQjwSeK-XO0aSca-rXy4jrv3pLJ3hx8cu-dyUwIOtZ4rKgDgNr1TLdYBXh-VAUusBL4vqgonFDNhxKNl8-KifAXqb-fLM7BnpQUqb8VMd91ht4_WE78jUNTLDW6jk_7JVNRhFxZUJ1a0w_9GBKEIYk8TX8Wxx52yEu8NhiRsAkAo5zDHA9guo9QyjxjU3kYhusUM3Ab5awOMMndZjkYRnzmXSgJLlqYZdjA8TJugEQgk7QBkYtpPRiYqaOHThxoq1H5dYkq9EC2USJzknOCI7Dtd5t-BgL9VoP2DwLKYLZgOGK9XankSk06sIFOvpiUGnOJPA")

# Where to download
download_dir = "mama_mia_dataset"
os.makedirs(download_dir, exist_ok=True)

# Recursive download function with tqdm
def download_entity(entity_id, location):
    children = syn.getChildren(entity_id)
    if not children:  # It's a file
        syn.get(entity_id, downloadLocation=location, ifcollision="overwrite.local")
        return 1  # count as 1 file downloaded
    else:  # It's a folder
        folder_name = syn.get(entity_id).name
        folder_path = os.path.join(location, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        count = 0
        for child in tqdm(children, desc=f"Downloading {folder_name}", unit="file"):
            count += download_entity(child['id'], folder_path)
        return count

# Start downloading
total_files = download_entity("syn60868042", download_dir)
print(f"Download complete! Total files downloaded: {total_files}")
