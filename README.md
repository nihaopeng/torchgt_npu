# torchgt on ascend npu

## ascend 910B4

## dataset

`python utils/preprocess_data.py [dataset_name](e.g. python utils/preprocess_data.py cora)`

## torch_npu

`bash run.sh origin 0,1 cora`

## install

install torch, torch_npu (租的服务器已经安装，1.11版本)

`pip install torch-geometric==2.0.0 torch-scatter==2.1.1 torch-summary==1.4.5 dgl==1.0.1 -i https://pypi.org/simple`

`pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html`

`pip install pymetis`

## tip
1, 当前版本请不要使用除了full以外的attn type，他们都没有返回score，也不被我们需要