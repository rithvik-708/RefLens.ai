#!/usr/bin/env bash
set -e

echo "[+] Initializing RefLens project workspace..."

# Create project skeleton
mkdir -p reflens/{configs,benchmarks,web}
mkdir -p reflens/dataset/{raw,incidents,splits}
mkdir -p reflens/src/{vision,detection,agent,backend,utils}

cd reflens

# Create Python environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r ../requirements.txt

# Create empty module indicators
touch src/__init__.py
touch src/vision/__init__.py
touch src/detection/__init__.py
touch src/agent/__init__.py
touch src/backend/__init__.py
touch src/utils/__init__.py

echo "[+] Directory structure created successfully."
