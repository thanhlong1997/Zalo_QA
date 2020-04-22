import random
import json
from sklearn.model_selection import train_test_split

with open('train.json','r',encoding='utf-8') as json_file:
    data = json.load(json_file)

# print(data[0])
# 1/0
valid = []
train_new = []
index = list(range(0,len(data)))
random.shuffle(index)
# print((index))
train_index = index[:int(len(index)*0.8)]
valid_index = index[int(len(index)*0.8)+1:]
# print(valid_index)
for x in valid_index:
    valid.append(data[x])

for x in train_index:
    train_new.append(data[x])
# print(train_new[0])

with open('train_new.json','w',encoding='utf-8') as file:
    json.dump(train_new,file)
with open('valid.json','w',encoding='uft-8') as file:
    json.dump(valid)