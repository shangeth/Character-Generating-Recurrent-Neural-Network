# Character-Generating-Recurrent-Neural-Network
``` git clone 
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

Give the path to save the model and change the hyper parameters as needed.
 ```
 python train.py --epoch=50 --batch_size=1500 --save_path='./trained_model/char_rnn_model.net'
 ```
```
Epoch: 1/50... Step: 10... Train Loss: 3.2590... Val Loss: 3.2764
Epoch: 2/50... Step: 20... Train Loss: 3.1488... Val Loss: 3.1495
Epoch: 3/50... Step: 30... Train Loss: 3.1222... Val Loss: 3.1219
Epoch: 4/50... Step: 40... Train Loss: 3.1175... Val Loss: 3.1158
Epoch: 5/50... Step: 50... Train Loss: 3.1105... Val Loss: 3.1127
Epoch: 6/50... Step: 60... Train Loss: 3.1084... Val Loss: 3.1109
Epoch: 7/50... Step: 70... Train Loss: 3.1076... Val Loss: 3.1095
Epoch: 8/50... Step: 80... Train Loss: 3.1051... Val Loss: 3.1082
Epoch: 9/50... Step: 90... Train Loss: 3.1013... Val Loss: 3.1065
Epoch: 10/50... Step: 100... Train Loss: 3.1173... Val Loss: 3.1037
...
...
```

## Generating Text from trained model
```
python generate_text.py -h
```
```
usage: generate_text.py [-h] [--model_path MODEL_PATH] [--num_chars NUM_CHARS]
                        [--prime PRIME] [--top_k TOP_K]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to trained model
  --num_chars NUM_CHARS
                        no of characters to generate
  --prime PRIME         Starting chars
  --top_k TOP_K         choose top k chars
```
