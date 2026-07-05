import urllib.request
import re
import os

def download_file_from_google_drive(file_id, destination):
    print(f"Starting download for file ID: {file_id}")
    url = "https://docs.google.com/uc?export=download"
    
    # Send an initial request to check for virus scan warnings/confirmation page
    req_url = f"{url}&id={file_id}"
    try:
        # Create a request with a standard user-agent to avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(req_url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            content = response.read()
            # If the response is small, check if it's an HTML confirmation page
            if len(content) < 100000:
                html_text = content.decode('utf-8', errors='ignore')
                match = re.search(r'confirm=([0-9A-Za-z_]+)', html_text)
                if match:
                    confirm_token = match.group(1)
                    print(f"Confirmation token found: {confirm_token}")
                    download_url = f"{url}&confirm={confirm_token}&id={file_id}"
                else:
                    download_url = req_url
            else:
                download_url = req_url
        
        # Download the file
        print("Downloading file content...")
        req = urllib.request.Request(download_url, headers=headers)
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            # Buffer the read to show progress
            total_size = int(response.headers.get('content-length', 0))
            bytes_downloaded = 0
            block_size = 1024 * 64  # 64 KB
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                bytes_downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = (bytes_downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({bytes_downloaded}/{total_size} bytes)", end='')
                else:
                    print(f"\rDownloaded: {bytes_downloaded} bytes", end='')
            print("\nDownload complete!")
            
    except Exception as e:
        print(f"\nError occurred: {e}")
        # Fallback to direct urlretrieve
        print("Attempting fallback direct download...")
        try:
            urllib.request.urlretrieve(f"{url}&id={file_id}", destination)
            print("Fallback download complete!")
        except Exception as e_fallback:
            print(f"Fallback failed: {e_fallback}")

if __name__ == "__main__":
    file_id = "1YVwuLd4zgrqM2i10XGf3OG2ean86RDuX"
    destination = "user_behavior_dataset.csv"
    download_file_from_google_drive(file_id, destination)
    
    # Check if the file exists and display its size
    if os.path.exists(destination):
        size = os.path.getsize(destination)
        print(f"Successfully downloaded '{destination}' ({size / (1024*1024):.2f} MB)")
    else:
        print("Download failed: File does not exist.")
