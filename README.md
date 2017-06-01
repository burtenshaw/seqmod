# encoder-decoder-torch

## Preprocessing

## Training 

### Basic Training 

```
python questions.py --path data/faq_0/train --target /home/burtenshaw/code/IMDB/seqmod/data/faq_0/val.questions --gpu 
```

### From Glove

```
python questions.py --path data/faq_0/train --target /home/burtenshaw/code/IMDB/seqmod/data/faq_0/val.questions --pretrained /home/burtenshaw/code/IMDB/seqmod/data/glove_0/ 
```

### Important Arguments 

* ```--gpu```
* ```--bidi```

### Example

```
python questions.py --path data/faq_0/train --gpu --target "apart from the old vhs by ifs that was later put on the video nasties list all bbfc 18 releases are modestly censored. approx. 90 seconds are missing and a detailed comparison between both versions with pictures can be found here" --pretrained /home/burtenshaw/code/IMDB/seqmod/data/glove_0/ --bidi --max_size 400000 --csv /home/burtenshaw/logging/testing_logger.csv --emb_dim 50 --layers 4
```
