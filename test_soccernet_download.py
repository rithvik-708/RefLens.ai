from SoccerNet.Downloader import SoccerNetDownloader
from pathlib import Path

GAME = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
OUTPUT = Path("dataset/raw/SoccerNet")

downloader = SoccerNetDownloader(
    LocalDirectory=str(OUTPUT)
)

downloader.password = "s0cc3rn3t"

print("Starting SoccerNet EXRCS download...")

downloader.downloadGame(
    files=["1_224p.mkv"],
    game=GAME,
    source="HuggingFace"
)

expected = OUTPUT / GAME / "1_224p.mkv"

print()
print("Expected file:")
print(expected)
print()

if expected.exists():
    print("SUCCESS")
    print(f"Size: {expected.stat().st_size / (1024**2):.2f} MB")
else:
    print("FAILED")
    print("File was not downloaded.")