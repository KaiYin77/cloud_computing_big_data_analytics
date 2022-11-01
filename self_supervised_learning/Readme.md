# Video_action_classification

## Dataset

- Unzip data.zip to `../data`

    ```sh
    unzip data.zip -d ../data
    ```

- Folder structure

    ```txt
    .
    ├── data/
    │   └── hw2/
    │   	├── test/
    │   	└── unlabeled/
    │ 
	└── self_supervised_learning 
    	├── models
    	│   └── resnet.py
		├── submit
    	│   └── xxxxx.npy
    	├── weights
    	│   └── xxxxx.ckpt
    	├── config.py
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
conda create --name ccbda_hw2 python=3.7
conda activate ccbda_hw2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

pip3 install -r requirements.txt
```


## Training Mode
__train from scratch__
```sh
python train.py --train
```

## Make inference for embedding

```sh
python train.py --test --ckpt xxxxx.ckpt 
```

