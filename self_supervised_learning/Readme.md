# Video_action_classification

## Dataset

- Unzip data.zip to `../data`

    ```sh
    unzip data.zip -d ../data
    ```

- Folder structure

    ```txt
    .
    ├── data
    │   ├── test/
    │   └── train/
    │ 
	└── self_supervised_learning 
		├── submit
    	│   └── xxxxx.npy
    	├── weights
    	│   └── xxxxx.ckpt
    	├── Readme.md
    	├── requirements.txt
    	├── dataloader.py
    	└── train.py
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
python train.py --train --net vgglstm
```

## Make Prediction

```sh
python train.py --test --net vgglstm --ckpt xxxxx.ckpt 
```
---

## Unit Test Pipeline - Dev mode

```sh
python train.py --dev --net vgglstm
```
## Unit Test Dataloader

```sh
python dataloader
```
## Validation Mode

```sh
python train.py --validate --net vgglstm --ckpt xxxxx.ckpt
```
