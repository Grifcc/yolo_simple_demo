# MLU Offline Examples: using Cambricon Neuware Runtime API

Cambricon Neuware Runtime (CNRT) provides the ability to run neural networks without the need of deep learning frameworks.

## environment
```
Ubuntu16.04
MLU270 / MLU220 M.2
```

## install
The first execution requires the installation of the runtime dependency library.
```bash
  sudo apt-get install libopencv-dev libgflags-dev libgoogle-glog-dev libboost-all-dev libgeos-dev zip python-opencv
  pip install Cython pycocotools numpy==1.17.0 matplotlib shapely pillow
```
