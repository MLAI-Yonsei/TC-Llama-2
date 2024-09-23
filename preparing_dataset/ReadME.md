# KISTI (전처리코드)

1. 전처리 코드는 ipynb파일로 구현되었습니다.
2.  collab에서 구동할 수 있는 코드로 만들어져 있습니다. 만일 collab을 사용하지 않는다면, 각 실행파일별로 맨 위의 use_collab=False로 바꾸어주세요.
3.  해당 kisti 전처리 폴더가 있는 file_path를 설정해주세요.
4. input_dataset은 해당 실행파일을 실행하기전에 필요한 데이터셋입니다.
5. output_dataset은 해당 실행파일을 실행한 후 생성되는 데이터셋입니다.
6. 실행 순서
    1. rawdata 폴더안에 들어갈 파일들을 준비해주세요.
    2. prompt_generating 폴더안의 실행파일을 아래 순서대로 실행해주세요
        1. 반드시 unspc_task_prompt.ipynb 를 실행 후 unspsc_translate_prompt.ipynb를 실행해주세요.
        2. 반드시 final_prompt_generate.ipynb를 맨 마지막에 실행해주세요.
    3. dataset 폴더안에 전처리된 파일이 모두 있는 것을 확인한 다음, kisti_final_data 폴더의 dataset_construct.ipynb 파일을 실행해주세요.
    4. Task1,2를 위한 데이터를 생성하기 위해 task12_generating 폴더 안의 실행파일을 아래 순서대로 실행해주세요.
        1.  task1_amazon상품_prompt.ipynb , task1_amazon카테고리_prompt.ipynb, task2_danawa상품_prompt.ipynb,task2_danawa카테고리_prompt.ipynb
    5. Task 3을 위한 데이터를 생성하기 위해 task3_generating 폴더 안의 실행파일을 아래 순서대로 실행해주세요.
        1. 아마존50_맵핑_prompt.ipynb, 다나와50_맵핑_prompt.ipynb, 조달청_맵핑_prompt.ipynb
    6. Task4(기술-사업화 관계성을 이용한 생성 task)를 생성하기 위해 NTISdata_generating폴더의 NTIS_C_generate.ipynb 파일을 실행해주세요.
   

## dataset

rawdata 폴더 안의 데이터를 이용하여 만들어진 데이터들이 저장되는 폴더입니다.

해당 폴더안의 파일들은 kisti_final_data폴더의 dataset_construct.ipynb에서 합쳐져 최종데이터가 됩니다.

### filelist

'조달청_task.csv'

'wipson_task.csv'

'unspsctask123.csv'

 'unspsc_task.csv'

'specprod_task.csv'

'kpa_task.csv'

'danawa_task.csv'

'amazon_task.csv'

'상품정보task.csv'

 'NTIS_task.csv'

## fine_tuning_data

### dataset_construct.ipynb

- input_dataset
    - /KISTI/fine_tuning_data/final_train_data.csv
    - /KISTI/fine_tuning_data/final_test_data.csv
- output_dataset
    - /KISTI/fine_tuning_data/kisti_train.json
    - /KISTI/fine_tuning_data/kisti_test.json

## prompt_generating

rawdata 폴더안의 데이터를 전처리하는 코드들이 있는 폴더입니다.

### zodalprompt.ipynb

- input_dataset
    - /KISTI/rawdata/조달청 분류 체계 전체자료.csv
    - /KISTI/rawdata/UNv240301.xlsx
- output_dataset
    - /KISTI/조달청_task.csv

### wipson_prompt.ipynb

- input_dataset
    - /KISTI/rawdata/wipson_total.csv
- output_dataset
    - /KISTI/dataset/wipson_task.csv

### unspc_task_prompt.ipynb

- input_dataset
    - /KISTI/rawdata/UNv240301.xlsx
    - /KISTI/rawdata/조달청 분류 체계 전체자료.csv
- output_dataset
    - /KISTI/dataset/unspsctask123.csv

### unspsc_translate_prompt.ipynb

- input_dataset
    - /KISTI/rawdata/UNv240301.xlsx
    - /KISTI/rawdata/조달청 분류 체계 전체자료.csv
    - "/KISTI/dataset/unspsctask123.csv
- output_dataset
    - /KISTI/dataset/unspsc_task.csv

### preprocessing_product.ipynb

- input_dataset
    - /KISTI/rawdata/상품정보 품목 등록 내역_제품특성정보소개추가.xlsx
- output_dataset
    - /KISTI/dataset/specprod_task.csv

### kpa_inst_data_preprocessing.ipynb

- input_dataset
    - /KISTI/rawdata/raw_dataset_kor.csv
- output_dataset
    - /KISTI/dataset/kpa_task.csv

### danawaprompt.ipynb

- input_dataset
    - /KISTI/rawdata/다나와_수집정보_20230328.xlsx
    - /KISTI/rawdata/danawa_data.xlsx
- output_dataset
    - /KISTI/rawdata/danawa12.csv
    - /KISTI/dataset/danawa_task.csv

### amazon_prompt.ipynb

- input_dataset
    - /KISTI/rawdata/아마존_수집정보_20230328 (1).xlsx
    - /KISTI/rawdata/amazon_data.xlsx
- output_dataset
    - /KISTI/dataset/amazon_task.csv

### 상품정보_prompt.ipynb

- input_dataset
    - /KISTI/rawdata/상품정보 품목 등록 내역.csv
- output_dataset
    - /KISTI/dataset/상품정보task.csv

### final_prompt_generate.ipynb

- input_dataset
    - /KISTI/dataset/NTIS_task.csv
    - /KISTI/dataset/조달청_task.csv
    - /KISTI/dataset/danawa_task.csv
    - /KISTI/dataset/amazon_task.csv
    - /KISTI/dataset/wipson_task.csv
    - /KISTI/dataset/kpa_task.csv
    - /KISTI/dataset/상품정보task.csv
    - /KISTI/dataset/specprod_task.csv
    - /KISTI/dataset/unspsc_task.csv
    - /KISTI/rawdata/다나와_아마존_조달청_맵핑.xlsx
- output_dataset
    - /KISTI/dataset/NTIS_task.csv
    - /KISTI/kisti_final_data/final_train_data.csv
    - "/KISTI/kisti_final_data/final_test_data.csv

## rawdata

전처리 전 파일들이 저장되는 폴더입니다.

### filelist

 '다나와_수집정보_20230328.xlsx'

 '아마존_수집정보_20230328 (1).xlsx'

 'danawa_data.xlsx'

 'wipson_total.csv'

 'wipson_data.csv'

 '다나와_아마존_조달청_맵핑.xlsx'

 'amazon_data.xlsx'

 '상품정보 품목 등록 내역.csv'

 '상품정보 품목 등록 내역_제품특성정보소개추가.xlsx'

 'UNv240301.xlsx'

 'danawa12.csv'

 'NTIS.csv'
``

## task1&2_generating

- Task 1 **: Amazon product to Amazon category mapping** 태스크를 실행하기 위해 데이터를 전처리하는 코드입니다

### task1_amazon상품_prompt.ipynb

- Amazon product prompt 생성
- input_dataset
    - /KISTI/kisti_final_data/final_test_data.csv
- output_dataset
    - /KISTI/task12_generating/amazon_product.csv
    - /KISTI/task12_generating/amazon_product.json
    

### task1_amazon카테고리_prompt.ipynb

- Amazon category prompt 생성
- input_dataset
    - /KISTI/dataset/amazon_task.csv
- output_dataset
    - /KISTI/task12_generating/amazon_category.csv
    - /KISTI/task12_generating/amazon_category.json

- Task 2:• **다나와 product to 다나와 category mapping** 태스크를 실행하기 위해 데이터를 전처리하는 코드입니다

### task2_danawa상품_prompt.ipynb

- danawa product prompt 생성
- input_dataset
    - /KISTI/kisti_final_data/final_test_data.csv
- output_dataset
    - /KISTI/task12_generating/danawa_product.csv
    - /KISTI/task12_generating/danawa_product.json

### task2_danawa카테고리_prompt.ipynb

- danawa category prompt 생성
- input_dataset
    - /KISTI/rawdata/다나와_수집정보_20230328.xlsx
- output_dataset
    - /KISTI/task12_generating/danawa_category.csv
    - /KISTI/task12_generating/danawa_category.json

### filelist

amazon_product.csv

amazon_product.json

amazon_category.csv

amazon_category.json

danawa_product.csv

danawa_product.json

danawa_category.csv

danawa_category.json

## task3_generating

- Task 3 : **Amazon / 다나와 product to 조달청 (표준분류) category mapping** 태스크를 실행하기 위해 데이터를 전처리하는 코드입니다
- 다나와_아마존_조달청_맵핑.xlsx는 다나와,아마존 데이터를 각각 50개씩 수집해서 조달청분류체계에 직접 매칭한 excel 파일입니다.
- 아마존50_맵핑_prompt.ipynb 실행 후 다나와50_맵핑_prompt.ipynb를 실행합니다

### 아마존50_맵핑_prompt.ipynb

- input_dataset
    - /KISTI/task3_generating/다나와_아마존_조달청_맵핑.xlsx
- output_dataset
    - /KISTI/task3_generating/amazon50.csv
    - /KISTI/task3_generating/amazon50.json

### 다나와50_맵핑_prompt.ipynb

- input_dataset
    - /KISTI/task3_generating/다나와_아마존_조달청_맵핑.xlsx
    - /KISTI/rawdata/danawa_data.xlsx
- output_dataset
    - /KISTI/task3_generating/danawa50.csv
    - /KISTI/task3_generating/danawa50.json

danawa50.csv, amazon50.csv 파일을 결합하여 최종 task3에서 사용할 100개의 test sample data를 생성합니다.

- input_dataset
    - KISTI/task3_generating/danawa50.csv
    - KISTI/task3_generating/amazon50.csv
- output_dataset
    - /KISTI/task3_generating/test100.csv
    - /KISTI/task3_generating/test100.json

### 조달청_맵핑_prompt.ipynb

- 조달청 분류체계별 prompt 생성
- Lv1, Lv2,Lv5에 대해 각각 별도의 데이터셋으로 생성하였습니다
- input_dataset
    - /KISTI/rawdata/조달청 분류 체계 전체자료.csv
- output_dataset
    - /KISTI/task3_generating/조달청_lv1.csv
    - /KISTI/task3_generating/조달청_lv1.json
    - /KISTI/task3_generating/조달청_lv2.csv
    - /KISTI/task3_generating/조달청_lv2.json
    - /KISTI/task3_generating/zodal.csv   (조달청 lv5 데이터에 해당)
    - /KISTI/task3_generating/zodal.json (조달청 lv5 데이터에 해당)

### filelist

amazon50.csv

amazon50.json

danawa50.csv

danawa50.json

test100.csv

test100.json

조달청_lv1.csv

조달청_lv1.json

조달청_lv2.csv

조달청_lv2.json

zodal.csv

zodal.json

## task4_generating

### task4_generate.ipynb

- Task4 : 기술-사업화 관계성을 이용한 생성 task 실행을 위한 데이터를 전처리하는 코드입니다.
- input_dataset
    - /KISTI/NTISdata_generating/NTIS_C_data.xlsx
- output_dataset
    - /KISTI/NTISdata_generating/kisti_taskXX.json

- 모델 실행 시 kisti_taskXX.json 파일을 input data로 설정하면 generated_predictions.jsonl 이 생성됩니다. (자세한 사항은 모델 실행 파트를 참고해 주십시오.)

### filelist

kisti_taskXX.json
generated_predictions.jsonl (실제로 이곳에 저장되는 파일은 아니이나, 참고를 위해 이곳에 저장하였습니다.)
