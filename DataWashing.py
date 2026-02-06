import pandas as pd
import warnings
import torch
import numpy as np
from gensim.models import Word2Vec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 消除警告
warnings.simplefilter(action='ignore', category=FutureWarning)

original_data = pd.read_excel("new_science_data.xlsx")

def one_hot_encode(column_name, prefix_name):
    """
    对指定列进行one-hot编码
    :param column_name: 列名
    :param prefix_name: one-hot编码后给列名添加前缀
    :return:
    """
    global original_data
    # 获取性别列的下标索引
    middle_index = list(original_data.columns).index(column_name)
    # 将数据分成前中后三部分，进行one-hot编码后再进行拼接
    front_data = original_data.iloc[:, :middle_index]
    middle_data = pd.get_dummies(original_data.iloc[:, middle_index], prefix=prefix_name, ).astype(int)
    behind_data = original_data.iloc[:, middle_index + 1:]
    original_data = pd.concat([front_data, middle_data, behind_data], axis=1)


def multi_one_hot_encode(column_name, prefix_name, separation):
    """
    对多标签列进行one-hot编码
    :param column_name: 列名
    :param prefix_name: one-hot编码后给列名添加前缀
    :param separation: 分隔符
    :return:
    """
    global original_data

    # 获取性别列的下标索引
    middle_index = list(original_data.columns).index(column_name)
    # 将数据分成前中后三部分，进行one-hot编码后再进行拼接
    front_data = original_data.iloc[:, :middle_index]
    middle_data = original_data[column_name].str.get_dummies(sep=separation)
    middle_data = middle_data.rename(columns=lambda x: f'{prefix_name}_{x}')
    behind_data = original_data.iloc[:, middle_index + 1:]
    original_data = pd.concat([front_data, middle_data, behind_data], axis=1)

# 4. 定义编码函数
def word2vec_encode(column_name, prefix_name, encode_dims):
    global original_data
    # 提取数据
    corpus = original_data[column_name].tolist()

    # 训练 Word2Vec 模型
    model = Word2Vec(sentences=corpus, vector_size=encode_dims, window=5, min_count=1, workers=8)
    # 获取每个单位名称的词向量（简单求平均作为单位名称的向量表示）
    unit_vectors = []
    for name in corpus:
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
    # 获取下标索引
    middle_index = list(original_data.columns).index(column_name)
    # 将数据分成前中后三部分，进行one-hot编码后再进行拼接
    front_data = original_data.iloc[:, :middle_index]
    unit_vectors_df = unit_vectors_df.rename(columns=lambda x: f'{prefix_name}_{x}')
    behind_data = original_data.iloc[:, middle_index + 1:]
    original_data = pd.concat([front_data, unit_vectors_df, behind_data], axis=1)


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

def rank_encode():
    """
    第四次清洗：对“等级”列进行编码，低等级为0，中等级为1，高等级为2
    :return:
    """
    global original_data
    col_index = list(original_data.columns).index("等级")
    original_data["等级"] = original_data.iloc[:, col_index]. \
        replace("低等级", 0). \
        replace("中等级", 1). \
        replace("高等级", 2)

def sex_encode():
    """
    第五次清洗：对“1.您的性别为”列进行编码，采用one-hot编码
    :return:
    """
    one_hot_encode('1.您的性别为', '性别')
    # word2vec_encode("1.您的性别为", "性别", 2)

def age_encode():
    """
    第六次清洗：对“2.您的年龄为”列进行编码，＜30岁为0，30-40岁为1，41-50岁为2，51岁及以上为3
    :return:
    """
    global original_data
    col_index = list(original_data.columns).index("2.您的年龄为")
    original_data = original_data.rename(columns={"2.您的年龄为": "年龄"})
    original_data["年龄"] = original_data.iloc[:, col_index]. \
        replace("＜30岁", 0). \
        replace("30-40岁", 1). \
        replace("41-50岁", 2). \
        replace("51岁及以上", 3)

def ethnic_encode():
    """
    第七次清洗：对“3.您的民族为”列进行Word2Vec编码，输出4维向量
    :return:
    """
    word2vec_encode("3.您的民族为", "民族", 4)


def degree_encode():
    """
    第八次清洗：对“4.您的最高学历（学位）”列进行编码，其他为0，本科为1，硕士研究生为2，博士研究生为3
    :return:
    """
    global original_data
    col_index = list(original_data.columns).index("4.您的最高学历（学位）")
    original_data = original_data.rename(columns={"4.您的最高学历（学位）": "最高学历"})
    original_data["最高学历"] = original_data.iloc[:, col_index]. \
        replace("其他", 0). \
        replace("本科", 1). \
        replace("硕士研究生", 2). \
        replace("博士研究生", 3)


def region_encode():
    """
    第九次清洗：对“5.您的所在地区”列进行编码，采用one-hot编码
    :return:
    """
    one_hot_encode('5.您的所在地区', '所在地区')


def work_place_encode():
    """
    第十次清洗：对“6.您的单位名称”列进行编码，采用Word2Vec编码，输出向量维度6维
    :return:
    """
    word2vec_encode("6.您的单位名称", "单位名称", 6)


def work_age_encode():
    """
    第十一次清洗：对“7.您的工作年限”列进行编码，＜5年为0，5-10年为1，11-20年为2，21-30年为3，31年及以上为4
    :return:
    """
    global original_data
    col_index = list(original_data.columns).index("7.您的工作年限")
    original_data = original_data.rename(columns={"7.您的工作年限": "工作年限"})
    original_data["工作年限"] = original_data.iloc[:, col_index]. \
        replace("＜5年", 0). \
        replace("5-10年", 1). \
        replace("11-20年", 2). \
        replace("21-30年", 3). \
        replace("31年及以上", 4)


def job_title_encode():
    """
    第十二次清洗：对“8.您的技术职称”列进行编码，其他为0，初级为1，中级为2，副高级为3，正高级为4
    :return:
    """
    global original_data
    col_index = list(original_data.columns).index("8.您的技术职称")
    original_data = original_data.rename(columns={"8.您的技术职称": "职称"})
    original_data["职称"] = original_data.iloc[:, col_index]. \
        replace("其他", 0). \
        replace("初级", 1). \
        replace("中级", 2). \
        replace("副高级", 3). \
        replace("正高级", 4)


def department_encode():
    """
    第十三次清洗：对“9.您所在的科室”列进行编码，采用Word2Vec编码,输出8维向量
    :return:
    """
    word2vec_encode("9.您所在的科室", "科室", 8)



def profession_encode():
    """
    第十四次清洗：对“ 10.您的所属专业？”列进行编码，采用Word2Vec编码，输出4维向量
    :return:
    """
    word2vec_encode(" 10.您的所属专业？", "所属专业", 4)



def scientific_activities_encode():
    """
    第十五次清洗：对“11.您从事科研活动的类型有哪些”列进行编码，采用Word2Vec编码，输出6维特征
    :return:
    """
    word2vec_encode("11.您从事科研活动的类型有哪些", "科研活动", 6)


def scientific_research_encode():
    """
    第十六次清洗：对“12.您从事科学研究的内容包括”列进行编码，采用Word2Vec编码，输出6维特征
    :return:
    """
    word2vec_encode("12.您从事科学研究的内容包括", "研究内容", 6)


def enhance_resource_encode():
    """
    第十七次清洗：对“19.您认为亟需强化的科研资源”列进行编码，采用Word2Vec编码，输出8维特征
    :return:
    """
    word2vec_encode("19.您认为亟需强化的科研资源", "需强化资源", 8)


def scientific_system_requirement():
    """
    第十八次清洗：对“20.您在科研管理系统方面的需求”列进行编码，采用Word2Vec编码，输出6维特征
    :return:
    """
    word2vec_encode("20.您在科研管理系统方面的需求", "科研系统需求", 6)


def scientific_info_acquire():
    """
    第十九次清洗：对“21.您需要获取科研信息的渠道”列进行编码，采用Word2Vec编码，输出6维特征
    :return:
    """
    word2vec_encode("21.您需要获取科研信息的渠道", "科研信息获取渠道", 6)


def scientific_training_frequency():
    """
    第二十次清洗：对“22.您接受的科研培训频率”列进行编码，采用one-hot编码
    :return:
    """

    multi_one_hot_encode("22.您接受的科研培训频率", "科研培训频率", "┋")


def policy_support():
    """
    第二十一次清洗：对“27.您需要获得哪些政策支持”列进行编码，采用Word2Vec编码，输出6维特征
    :return:
    """
    word2vec_encode("27.您需要获得哪些政策支持", "政策支持", 6)



def professor_enhance():
    """
    第二十二次清洗：对“28.援疆专家对当地医务人员科研能力提升的作用”列进行编码，采用Word2Vec编码，输出6维特征
    :return:
    """
    word2vec_encode("28.援疆专家对当地医务人员科研能力提升的作用", "专家对医务人员科研能力提升", 6)


def delete_last_two_cols():
    """
    第二十三次清洗：删除最后两列（29.您的其他科研需求）、（30.您对医院构建"临床-科研"双向促进机制的具体建议）
    :return:
    """
    global original_data
    original_data = original_data.iloc[:, :-2]

if __name__ == '__main__':
    zero_padding()
    float_to_int()
    exchange_error_data()
    rank_encode()
    sex_encode()
    age_encode()
    ethnic_encode()
    degree_encode()
    region_encode()
    work_place_encode()
    work_age_encode()
    job_title_encode()
    department_encode()
    profession_encode()
    scientific_activities_encode()
    scientific_research_encode()
    enhance_resource_encode()
    scientific_system_requirement()
    scientific_info_acquire()
    scientific_training_frequency()
    policy_support()
    professor_enhance()
    delete_last_two_cols()
    original_data.to_excel('./temp_data/23-data.xlsx', index=False)

