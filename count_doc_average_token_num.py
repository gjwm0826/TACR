import json
import statistics
import pandas as pd
from org.myrag.summary.config import *
from transformers import BartTokenizer
from org.myrag.summary.data import *

if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained(BART_PATH)
    # 原有39466篇原文
    json_datas = pd.read_json(PROJECT_BASE_PATH + '/org/myrag/summary/abstractive_train_subsampling.json', lines=True)
    print(len(json_datas))
    # 剔除summary为“ ”的后有29681篇原文
    # json_datas = json_datas[json_datas['summary'] != ' ']
    # print(len(json_datas))
    # 最大2105
    passage_length_list = []
    # extractive_summary_length_list = []
    # 最大178
    # summary_length_list = []
    max_token_len = 1
    min_token_len = 10000
    # max_sents_len = 1
    # min_sents_len = 10000
    for index, json_data in json_datas.iterrows():
        # dct = tokenizer.encode_plus(json_data['passages'], return_tensors='pt', add_special_tokens=False, padding=False)
        if len(json_data['passages']) > max_token_len:
            max_token_len = len(json_data['passages'])
        if len(json_data['passages']) < min_token_len:
            min_token_len = len(json_data['passages'])
        # seg = get_segmentation(json_data['passages'])
        # if len(seg) > max_sents_len:
        #     max_sents_len = len(seg)
        # if len(seg) < min_sents_len:
        #     min_sents_len = len(seg)
        passage_length_list.append(len(json_data['passages']))
        # abstractive_ans = tokenizer.encode(json_data['extractive_summary'], return_tensors="pt")
        # extractive_summary_length_list.append(extractive_ans.shape[1])
        # extractive_ans = len(json_data['summary'])
        # if extractive_ans > max_len:
        #     max_len = extractive_ans
        # ans = tokenizer.encode_plus(json_data['summary'], return_tensors="pt", add_special_tokens=False, padding=False)
        # summary_length_list.append(ans['input_ids'].shape[1])
    # print(max_len)
    # 计算25分位数
    passage_quantile_25 = statistics.quantiles(passage_length_list, n=100)[25]
    # extractive_summary_quantile_95 = statistics.quantiles(extractive_summary_length_list, n=100)[100]
    # summary_quantile_95 = statistics.quantiles(summary_length_list, n=100)[95]

    # 训练集95分位733，验证集738，测试集737
    print('段落长度45分位：' + str(passage_quantile_25))
    #
    # # 82
    # print('提取摘要长度95分位：' + str(extractive_summary_quantile_95))
    #
    # 训练集和验证集95分位83，测试集82
    # print('摘要长度95分位：' + str(summary_quantile_95))

    print('最大token数量：' + str(max_token_len))
    print('最小token数量：' + str(min_token_len))
    print('最大句子数量：' + str(max_sents_len))
    print('最小句子数量：' + str(min_sents_len))