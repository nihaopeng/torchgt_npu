# torchgt on ascend npu

## ascend 910B4

## run

torchgt

cd TorchGT

bash scripts/1_efficiency.sh

## torch_npu

bash run.sh

## install

install torch, torch_npu (租的服务器已经安装，1.11版本)

pip install torch-geometric==2.0.0 torch-scatter==2.1.1 torch-summary==1.4.5 -i https://pypi.org/simple

pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html