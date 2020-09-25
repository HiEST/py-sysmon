#!/bin/bash

# Dependencies: pcm and py-rapl
mkdir build
cd build/
git clone https://github.com/opcm/pcm
cd pcm/
make pcm.x -j
cd ../

git clone https://github.com/wkatsak/py-rapl.git
cd py-rapl/
pip install -e .
cd ../../

cp build/pcm/pcm.x utils/
