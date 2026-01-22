"""
Download NeuroScan VGG19 Model from HuggingFace
For Windows Local Development
"""

import os
import requests
from pathlib import Path
import sys

# Configuration
MODEL_URL = "https://huggingface.co/Sankalpadhikari/vgg19_final_20260110_154609/resolve/main/vgg19_final_20260110_154609.pth"
MODEL_PATH = Path(__file__).parent / "models" / "vgg19.pth"


def download_with_progress(url, destination):
    """Download file with progress bar"""

    # Create directory
    os.makedirs(destination.parent, exist_ok=True)

    # Check if already exists
    if destination.exists():
        print(f"✅ Model already exists at {destination}")
        response = input("Download again? (y/n): ")
        if response.lower() != 'y':
            return
        os.remove(destination)

    print(f"📥 Downloading model from HuggingFace...")
    print(f"URL: {url}")
    print(f"Destination: {destination}")
    print()

    try:
        # Start download
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        total_mb = total_size / (1024 * 1024)

        print(f"File size: {total_mb:.1f} MB")
        print("Downloading...")
        print()

        downloaded = 0

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Show progress
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        downloaded_mb = downloaded / (1024 * 1024)

                        # Update progress bar
                        bar_length = 50
                        filled = int(bar_length * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_length - filled)

                        print(f'\r[{bar}] {progress:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)', end='')

        print()  # New line after progress
        print()
        print(f"✅ Model downloaded successfully!")
        print(f"📁 Location: {destination}")
        print(f"📊 Size: {os.path.getsize(destination) / (1024 * 1024):.1f} MB")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        if destination.exists():
            os.remove(destination)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Download cancelled")
        if destination.exists():
            os.remove(destination)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if destination.exists():
            os.remove(destination)
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroScan Model Downloader")
    print("=" * 60)
    print()

    download_with_progress(MODEL_URL, MODEL_PATH)

    print()
    print("=" * 60)
    print("Ready to use! You can now run: python app.py")
    print("=" * 60)