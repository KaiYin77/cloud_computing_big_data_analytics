# Diffusion_generative_model

## Dataset

- Unzip data.zip to `../data`

    ```sh
    unzip data.zip -d ../data
    ```

- Folder structure

    ```txt
    .
    ├── data/
    │   └── hw3/
    │   	└── mnist/
    │ 
	└── diffusion_generative_model 
    	├── models
    	│   └── resnet.py
    	├── weights
    	│   └── xxxxx.ckpt
    	├── config.yaml
    	├── criterion.py
    	├── dataloader.py
    	├── Readme.md
    	├── requirements.txt
    	├── train.py
    	└── utils.py
    ```
## Environment
Boosted by pytorch-lightning

### Create Environment

```sh
conda create --name ccbda_hw3 python=3.7
conda activate ccbda_hw3

pip3 install -r requirements.txt
```


## Training Mode
__train from scratch__
```sh
python train.py [--train] [--ckpt xxx.ckpt]
```
- `--train` : train mode
- `--ckpt CKPT` : ckpt filename 

## Make inference for embedding

```sh
python train.py --test --ckpt xxxxx.ckpt 
```

