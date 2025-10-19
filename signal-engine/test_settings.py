#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Check environment
print("=== Environment Variables ===")
print(f"SYMBOLS={repr(os.environ.get('SYMBOLS', 'NOT SET'))}")
print(f"symbols={repr(os.environ.get('symbols', 'NOT SET'))}")

print("\n=== Checking .env file ===")
env_file = Path(__file__).parent / ".env"
print(f".env exists: {env_file.exists()}")
if env_file.exists():
    content = env_file.read_text()
    for i, line in enumerate(content.split("\n"), 1):
        if "SYMBOLS" in line.upper():
            print(f"Line {i}: {repr(line)}")

print("\n=== Loading Settings ===")
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("Pydantic settings loaded successfully")

    from config import Settings

    settings = Settings()
    print(f"SUCCESS! Symbols: {settings.symbols}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
