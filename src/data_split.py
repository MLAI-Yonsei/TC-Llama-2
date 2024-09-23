
import json
import os
import random
from collections import defaultdict

def split_dataset(input_json, output_dir, val_ratio, random_seed):
    random.seed(random_seed)

    with open(input_json) as json_reader:
        dataset = json.load(json_reader)

    random.shuffle(dataset)
    
    num_val = int(len(dataset) * val_ratio)
    num_train = len(dataset) - num_val

    val_data, train_data = dataset[:num_val], dataset[num_val:]

    output_train_json = os.path.join(output_dir, 'train.json')
    output_val_json = os.path.join(output_dir, 'val.json')

    print(f'write {output_train_json}')
    with open(output_train_json, 'w') as train_writer:
        json.dump(train_data, train_writer)

    print(f'write {output_val_json}')
    with open(output_val_json, 'w') as val_writer:
        json.dump(val_data, val_writer)
        

def split_dataset_(input_json, output_dir, val_ratio, random_seed):
    random.seed(random_seed)
    
    with open(input_json) as json_reader:
        dataset = json.load(json_reader)
    random.shuffle(dataset)
    # 데이터를 category를 기준으로 그룹화하기 위한 빈 딕셔너리 초기화

    t = set()
    # 데이터를 category를 기준으로 그룹화
    for data in dataset:
        category = data['meta']['category']
        t.add(category)
    
    train_data = []
    val_data = []
    
    # 각 그룹별로 데이터를 섞음
    for i in t:
        tas = []
        for data in dataset:
            if data['meta']['category'] == i:
                tas.append(data)
        num_val = int(len(tas) * val_ratio)
        val_data += tas[:num_val]
        train_data += tas[num_val:]
        
    train_data += val_data

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 훈련 데이터 저장
    output_train_json = os.path.join(output_dir, 'kisti_train.json')
    print(f'write {output_train_json}')
    with open(output_train_json, 'w') as train_writer:
        json.dump(train_data, train_writer)
        
#input_json, output_dir, val_ratio, random_seed
input = '/data1/shared_dataset/kisti_instruction/kisti_train.json' #input 경로
output_dir = '/data1/yewon/LLaMA-Efficient-Tuning/data' #output 경로
split_dataset_(input, output_dir, 0.1 ,10)