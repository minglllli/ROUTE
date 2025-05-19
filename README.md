## ROUTE code

# Requirements
Firstly, please install the required environment for ROUTE.
```
Python 3.8
PyTorch 2.0
Torchvision 0.15
Tensorboard
```
Other dependencies are listed in requirements.txt.
To install requirements, run:

```
conda create -n route python=3.8 -y
conda activate route
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard
pip install -r requirements.txt
```
Please install the dependencies with the following versions.
```
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.2.1
yacs==0.1.8
tqdm==4.64.1
ftfy==6.1.1
regex==2022.7.9
timm==0.6.12
```

After installing the environment, to run the code

Change the root in config file ROUTE-code_submit/configs/data/cifar100_set1_2sample_40.yaml to path/to/your/dataset.

Then run the comamnd below:
```
python main.py -d cifar100_set1_2sample_40 -m vpt_vit_vit_b16_0
```
