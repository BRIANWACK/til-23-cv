#!/bin/sh
if ! pwd | grep -q "noteable"; then
    echo "Not on competition platform, exiting..."
    exit 0
fi
echo "Setting up for competition platform..."

echo "pip install -qr requirements.txt"
if pip install -qr requirements.txt; then
    echo "Successfully installed requirements.txt"
fi

echo "Symlink dataset folders"
echo "ln -sf ../datasets/Storage ./runs"
ln -sf ../datasets/Storage ./runs
echo "ln -sf ../datasets/models ./models"
ln -sf ../datasets/models ./models
echo "ln -sf ../datasets/data ./data"
ln -sf ../datasets/data ./data

echo "Setup complete!"
