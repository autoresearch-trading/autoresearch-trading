#!/usr/bin/env python3
from pathlib import Path
from dotenv import dotenv_values

env_file = Path(".env")
print(f"Loading: {env_file.absolute()}")
print(f"Exists: {env_file.exists()}")

values = dotenv_values(env_file)
print(f"\nLoaded {len(values)} values:")
for key, value in sorted(values.items()):
    if 'SYMBOL' in key.upper():
        print(f"  {key}={repr(value)}")

print(f"\nAll keys:")
for key in sorted(values.keys()):
    print(f"  - {key}")

# Check for lowercase symbols
if 'symbols' in values:
    print(f"\n⚠️  Found lowercase 'symbols': {repr(values['symbols'])}")
if 'SYMBOLS' in values:
    print(f"\n✅ Found uppercase 'SYMBOLS': {repr(values['SYMBOLS'])}")

