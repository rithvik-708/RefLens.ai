from SoccerNet.Downloader import SoccerNetDownloader
from pathlib import Path

GAME = (
    "england_epl/2014-2015/"
    "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
)

OUTPUT = Path("dataset/raw/SoccerNet")

downloader = SoccerNetDownloader(
    LocalDirectory=str(OUTPUT)
)

# TEMPORARY TEST ONLY
downloader.password = "s0cc3rn3t"

print("Starting SoccerNet EXRCS download...")
print(f"Game: {GAME}")
print("File: 1_720p.mkv")

downloader.downloadGame(
    files=["1_720p.mkv"],
    game=GAME,
    source="EXRCSDrive",
)

expected = OUTPUT / GAME / "1_720p.mkv"

print("\n--------------------------------")
print("RESULT")
print("--------------------------------")

if expected.exists() and expected.stat().st_size > 0:
    print("SUCCESS!")
    print(f"File: {expected}")
    print(
        f"Size: "
        f"{expected.stat().st_size / (1024 ** 2):.2f} MB"
    )
else:
    print("FAILED")
    print(f"Expected file: {expected}")