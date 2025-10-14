#!/usr/bin/env python3
"""Setup QuestDB locally without Docker - for development only."""

import subprocess
import sys
import time
from pathlib import Path

def check_questdb_installed():
    """Check if QuestDB is installed locally."""
    try:
        result = subprocess.run(['questdb', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_questdb():
    """Install QuestDB using Homebrew."""
    print("Installing QuestDB...")
    try:
        subprocess.run(['brew', 'install', 'questdb'], check=True)
        print("✓ QuestDB installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install QuestDB")
        return False

def start_questdb():
    """Start QuestDB service."""
    print("Starting QuestDB...")
    try:
        # Start QuestDB in background
        subprocess.Popen(['questdb', 'start'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)  # Wait for QuestDB to start
        
        # Check if it's running
        result = subprocess.run(['questdb', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ QuestDB started successfully")
            return True
        else:
            print("✗ QuestDB failed to start")
            return False
    except Exception as e:
        print(f"✗ Error starting QuestDB: {e}")
        return False

def setup_schema():
    """Setup QuestDB schema."""
    print("Setting up QuestDB schema...")
    try:
        subprocess.run([sys.executable, 'scripts/setup_questdb.py'], check=True)
        print("✓ QuestDB schema created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to setup schema: {e}")
        return False

def main():
    """Main setup function."""
    print("Setting up QuestDB for local development...")
    
    # Check if QuestDB is installed
    if not check_questdb_installed():
        print("QuestDB not found. Installing...")
        if not install_questdb():
            print("Please install QuestDB manually: brew install questdb")
            return False
    
    # Start QuestDB
    if not start_questdb():
        print("Please start QuestDB manually: questdb start")
        return False
    
    # Setup schema
    if not setup_schema():
        print("Please setup schema manually: python scripts/setup_questdb.py")
        return False
    
    print("\n🎉 QuestDB setup complete!")
    print("QuestDB is running on:")
    print("  - HTTP: http://localhost:9000")
    print("  - PostgreSQL: localhost:8812")
    print("\nYou can now run:")
    print("  - make signal-pipeline")
    print("  - make test")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
