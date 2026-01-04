import subprocess
import time
from pathlib import Path

SEGMENTED_DIR = Path("Samples/Segmented")

for speaker_dir in sorted(SEGMENTED_DIR.iterdir()):
    if not speaker_dir.is_dir():
        continue

    print(f"\n=== {speaker_dir.name} ===")

    for segment in sorted(speaker_dir.glob("*.wav")):
        print(f"  Playing: {segment.name}")
        subprocess.run(["afplay", str(segment)], check=True)
        time.sleep(0.5)

    input("Press Enter to continue to next speaker...")
