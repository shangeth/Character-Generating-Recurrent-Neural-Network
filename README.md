# Character-Generating-Recurrent-Neural-Network

## Training the model
Place the text corpus as data/data.txt

```python train.py -h```
```

usage: train.py [-h] [--data_preprocess DATA_PREPROCESS]
                [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                [--seq_len SEQ_LEN] [--lr LR] [--print_every PRINT_EVERY]
                [--save_path SAVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --data_preprocess DATA_PREPROCESS
                        Preprocess the data?
  --batch_size BATCH_SIZE
                        no of sequence in a batch
  --epochs EPOCHS       no of epochs to train
  --seq_len SEQ_LEN     no of character in a sequence
  --lr LR               learning rate
  --print_every PRINT_EVERY
                        show loss every n steps
  --save_path SAVE_PATH
                        path to save the model
```
