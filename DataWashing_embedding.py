import pandas as pd
import warnings
import torch
import numpy as np
from gensim.models import Word2Vec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 消除警告
warnings.simplefilter(action='ignore', category=FutureWarning)

original_data = pd.read_excel("new_science_data.xlsx")

# 4. 定义编码函数
def word2vec_encode(sentence, prefix_name, encode_dims):

    # 训练 Word2Vec 模型
    model = Word2Vec(sentences=sentence, vector_size=encode_dims, window=15, min_count=1, workers=8)
    # 获取每个单位名称的词向量（简单求平均作为单位名称的向量表示）
    unit_vectors = []
    for name in sentence:
        if name:
            vectors = [model.wv[word] for word in name if word in model.wv]
            if vectors:
                unit_vector = np.mean(vectors, axis=0)
                unit_vectors.append(unit_vector)
            else:
                unit_vectors.append(np.zeros(encode_dims))
        else:
            unit_vectors.append(np.zeros(encode_dims))

    # 将结果转换为 DataFrame
    unit_vectors_df = pd.DataFrame(unit_vectors)
    return unit_vectors_df.rename(columns=lambda x: f'{prefix_name}_{x}')


def zero_padding():
    """
    第一次清洗，将为空的数据进行零填充
    :return:
    """
    global original_data
    original_data = original_data.fillna(0)

def float_to_int():
    """
    第二次清洗，将小数四舍五入为整数
    :return:
    """
    global original_data
    # 1. 筛选所有 float 类型的列
    float_cols = original_data.select_dtypes(include='float64').columns

    # 2. 对 float 列四舍五入后转为 int
    original_data[float_cols] = original_data[float_cols].round(0).astype(int)

def exchange_error_data():
    """
    第三次清洗：替换异常的数据，评分部分数据应该在0到5之间，部分评分数据超出了5，超出5的数据全部按最高分数5分来赋分
    :return:
    """
    global original_data
    # 筛选出下标为33的列及其之后的所有整数列(下标为33的列及其之后的整数列均为评分数据)
    int_cols = original_data[original_data.columns[33:]].select_dtypes(include='int32').columns
    original_data[int_cols] = original_data[int_cols].map(lambda x: x if x <= 5 else 5)


def multi_word2vec_encode():
    """
    合并merge_column_name中指定的列进行合并(merge_column_name中最后两列不进行合并)，以逗号作为分隔符，合并后使用Word2Vec进行编码生成100维的向量
    :return:
    """
    global original_data
    merge_column_name = ['等级', '1.您的性别为', '2.您的年龄为', '3.您的民族为', '4.您的最高学历（学位）', '5.您的所在地区',
                         '6.您的单位名称', '7.您的工作年限', '8.您的技术职称', '9.您所在的科室', ' 10.您的所属专业？',
                         '11.您从事科研活动的类型有哪些', '12.您从事科学研究的内容包括', '19.您认为亟需强化的科研资源',
                         '20.您在科研管理系统方面的需求', '21.您需要获取科研信息的渠道', '22.您接受的科研培训频率',
                         '27.您需要获得哪些政策支持', '28.援疆专家对当地医务人员科研能力提升的作用', '29.您的其他科研需求',
                         '30.您对医院构建"临床-科研"双向促进机制的具体建议']
    temp_original_data = original_data[merge_column_name[:-2]].astype(str).apply(','.join, axis=1).tolist()
    temp_original_data = word2vec_encode(temp_original_data, "混合列", 100)
    original_data = original_data.drop(columns=merge_column_name)
    original_data = pd.concat([original_data, temp_original_data], axis=1)





if __name__ == '__main__':
    zero_padding()
    float_to_int()
    exchange_error_data()
    multi_word2vec_encode()
    original_data.to_excel('./temp_data/23-1-data.xlsx', index=False)