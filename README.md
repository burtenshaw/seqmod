# encoder-decoder-torch

## Preprocessing

## Training 

### Basic Training 

```
python simplify.py --path data/faq_0/train --target /home/burtenshaw/code/IMDB/seqmod/data/faq_0/val.questions --gpu 
```

### From Glove

```
python simplify.py --path data/faq_0/train --target /home/burtenshaw/code/IMDB/seqmod/data/faq_0/val.questions --pretrained /home/burtenshaw/code/IMDB/seqmod/data/glove_0/ 
```

### Important Arguments 

* ```--gpu```
* ```--bidi```
* 

### Example

```
python simplify.py --path data/faq_0/train --gpu --target /home/burtenshaw/code/IMDB/seqmod/data/faq_0/val.answers --pretrained /home/burtenshaw/code/IMDB/seqmod/data/glove_0/ --bidi --max_size 400000 --logging --emb_dim 50
```
