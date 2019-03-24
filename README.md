# Character-Generating-Recurrent-Neural-Network
```
git clone https://github.com/shangeth/Character-Generating-Recurrent-Neural-Network.git 
cd Character-Generating-Recurrent-Neural-Network
```
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

```
!python generate_text.py --model_path='./trained_model/char_rnn_model.net' --num_chars=1000 --prime='The'
```
### after 5 epochs
```
He  t ee a oee  ae  te  ea toee a ttoe oeaea oo aa    ot a e eet tte e      eete  oaa ateeeeta eeete eoaaaeoeoe  t e ae eet eo tteaoeoeeat   tote eetooetotao teooeet aaaoo a ao ettot tooto   e tet oae too e t aaoeo  t o   oa a eteaate ee t oe eet a toe  eo teot eaet   a    tta at ee t otot   ee oeoeeoe aao aa o eeeto e eaeeea aoa  eeotaa tooato ee otaatot aaaoeoeoo te eto t etoete    aatott ttt    toat ee    oo te eoe eeet a e  taateaee aeetoeeooe aa ee aette  ee ete e eeoto  aetta   o a e  t ttea ae t eoo  teeoeoaoa t a otteaot te a    tao  aeattateaeoet tta  ea a ao o o ae   eto eta aeeteeaao ot o oeo aet  e  a o o  ao atot   o  ae e e t   oeetae  otoae  a etttota  te o otet    to eeet aataoeaeoe a tae aetto a e eaoeee  e eaee eeo te ee  ttt o  t  oo o aae a   teate aaeea  oo  e e eooo o o ota tt tot t  tae eaaeee eee  e ote ot oa   te eote to  ea oo eo  aa eo  o tt o   o  o e ea aoeae eeae  aa  e oeea  eoe et  o ae e t to e   te aoeat  e ae e otaaaa teeot ee  e tat eoao    e   oe a ooo
```

### after 50 epochs
```
He to said, and wer his a meante tild.

"Whet the she shower the wout what hus as she manter and with,
stal is attere to
whe seres ald as as that her hent of a man she stouls his her and
the stist to him homs a sont her she sharsion. The would thather the
shing the selt, ans and, she
ser ting werl allan at his tail stone withe to was see with whis sains of hered him hans whe ser him his," the
said theighal to her working her that sompied of whe meching him him as he was, was stoung and to shis had his was sould the she that his hainss that his stilk, but
shours the said, the mering all with stard as that sele to he have a said when her treed."

"I his son wond ther
sompision tome with wish a sens the wout, but
the pass was
hersend, his
show in that her he have the pelant thuugh her,
and at his'ly what to bot to le ser the was han stould all he soren was that han son what to that said was so the parster was tond, and she sare sead.... I shit an she maness hom he shat to he wams him song a p
```
