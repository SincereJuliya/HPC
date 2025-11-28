#!/bin/bash

echo "==== Building project ===="
mkdir -p build
cd build
cmake ..
make

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "==== Running spectral clustering ===="
./spectral

if [ $? -ne 0 ]; then
    echo "Program failed!"
    exit 1
fi

cd ..

echo ""
echo "==== Running visualization ===="
python3 scripts/visualize.py \
    --data scripts/data/mixed_dataset.csv \
    --labels data/mixed_dataset_labels.csv \
    --output plots/nonmpi_visualization.png
