## Quick Start


## Install PLUME via pip:
```
git clone 
cd PLUME/
mkdir -p dataset
wget -O dataset/longlamp_abstract_generation.zip "https://drive.usercontent.google.com/download?id=1eKPqAiV7dK2lCZ0D6SmorERtce9AXgn8&export=download&authuser=0"
cd dataset
unzip longlamp_abstract_generation.zip
cd ..
conda create -n plume python=3.10
conda activate plume 
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install pytorch==2.4.0 torchvision=0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Generate the Base Fine-tuned LoRA Model
```
sh scripts/<taskname>/run_lora.sh
```

## Run Personalize Training of PLUME or PLUME-s
```
sh scripts/<taskname>/run_plume(_share).sh
```