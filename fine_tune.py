#from openprompt import
from sklearn.cluster import k_means
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, DBpediaProcessor, ImdbProcessor, \
    AmazonProcessor
#from openprompt.data_utils import InputExample
import argparse
import umap
import sklearn.metrics as sm
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import  rgb2hex
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str, default="new_snippets")
parser.add_argument("--cos_sim_num", type=int, default=0.3)
parser.add_argument("--score_limit", type=int, default=0.3)
parser.add_argument("--k", type=int, default=8)
parser.add_argument("--is_pre", type=bool, default=True)
parser.add_argument("--re_custom_model", type=bool, default=False)
parser.add_argument("--use_custom_model", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--umap", type=bool, default=True)
parser.add_argument("--verbalizer", type=str, default="kpt")
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--top_sentences", default=20, type=int)
parser.add_argument("--fine_tune_epoch", default=10, type=int)
args = parser.parse_args()
import copy

def hungarian_cluster_acc(x, y):
    assert x.shape == y.shape
    #assert x.min() == 0
    #assert y.min() == 0

    m = 1 + max(x.max(), y.max())
    n = len(x)
    total = np.zeros([m, m])
    for i in range(n):
        total[x[i], int(y[i])] += 1
    w = total.max() - total
    w = w - w.min(axis=1).reshape(-1, 1)
    w = w - w.min(axis=0).reshape(1, -1)
    while True:
        picked_axis0 = []
        picked_axis1 = []
        zerocnt = np.concatenate([(w == 0).sum(axis=1), (w == 0).sum(axis=0)], axis=0)

        while zerocnt.max() > 0:

            maxindex = zerocnt.argmax()
            if maxindex < m:
                picked_axis0.append(maxindex)
                zerocnt[np.argwhere(w[maxindex, :] == 0).squeeze(1) + m] = \
                    np.maximum(zerocnt[np.argwhere(w[maxindex, :] == 0).squeeze(1) + m] - 1, 0)
                zerocnt[maxindex] = 0
            else:
                picked_axis1.append(maxindex - m)
                zerocnt[np.argwhere(w[:, maxindex - m] == 0).squeeze(1)] = \
                    np.maximum(zerocnt[np.argwhere(w[:, maxindex - m] == 0).squeeze(1)] - 1, 0)
                zerocnt[maxindex] = 0
        if len(picked_axis0) + len(picked_axis1) < m:
            left_axis0 = list(set(list(range(m))) - set(list(picked_axis0)))
            left_axis1 = list(set(list(range(m))) - set(list(picked_axis1)))
            delta = w[left_axis0, :][:, left_axis1].min()
            w[left_axis0, :] -= delta
            w[:, picked_axis1] += delta
        else:
            break
    pos = []
    for i in range(m):
        pos.append(list(np.argwhere(w[i, :] == 0).squeeze(1)))

    def search(layer, path):
        if len(path) == m:
            return path
        else:
            for i in pos[layer]:
                if i not in path:
                    newpath = copy.deepcopy(path)
                    newpath.append(i)
                    ans = search(layer + 1, newpath)
                    if ans is not None:
                        return ans
            return None

    path = search(0, [])
    totalcorrect = 0
    for i, j in enumerate(path):
        totalcorrect += total[i, j]
    return totalcorrect / n


def read_data(fileHandler,fileWriter):
    count = 0
    line_list = []
    label_list = []
    while True:
        line = fileHandler.readline()
        if not line:
            return count, line_list,label_list
        else:
            #fileWriter.write(line)
            #对line的处理
            if args.dataset == "agnews":
                line = line.replace("\n","")
                label_and_txt = line.split(",")
                label_list.append(int(label_and_txt[0].replace("\"","")))
                line_list.append(label_and_txt[1].replace("\"",""))
            else:
                line = line.replace("\n", "")
                label_and_txt = line.split(",")
                label_list.append(int(label_and_txt[0].replace("\"", "")))
                line_list.append(label_and_txt[1].replace("\"", ""))
            #line_list.append(label_and_txt[1])
            count += 1

def sort_sentences(result_layer,line_list):
    sorted_result = [[] for _ in range(args.k)]
    count = 0
    for item in result_layer:
        #print("================================")
        #print(item)
        sorted_result[int(item)].append([int(item),line_list[int(count)]])
        count += 1
    return sorted_result

def save_cluster_sentence(data,path):
    f_write = open(path,"w")
    for item in data:
        for it in item:
            #print(it[1])
            f_write.write(str(it[0])+" "+it[1] + '\n')
    f_write.close()

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "./model/bert-base-cased")

from openprompt.prompts import ManualTemplate
promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It topic was about {"mask"}',
    tokenizer = tokenizer,
)

from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch
import numpy as np

def prompt_classification_unsupervised(sentences):
    classes = get_classes()

    cutoff = 0.5

    if args.verbalizer == "kpt":
        myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=classes, candidate_frac=cutoff,
                                               max_token_split=args.max_token_split).from_file(
            f"./scripts/cluster/{args.dataset}/knowledgeable_verbalizer.txt")
    elif args.verbalizer == "manual":
        myverbalizer = ManualVerbalizer(tokenizer, classes=classes).from_file(
            f"./scripts/cluster/{args.dataset}/manual_verbalizer.txt")
    else :
        raise NotImplementedError

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=myverbalizer,
    )

    result_matrix = [[0 for _ in range(args.k)] for z in range(args.k)]
    count_line = 0

    for item in sentences:
        datasets = []
        for it in item:
            label_index = it[0]
            text_a = it[1]
            dataset = InputExample(guid=str(label_index), text_a=text_a, label=int(label_index))
            datasets.append(dataset)
        # datasets是一个质心所有的句子
        # 每个句子跑prompt 多数投票 选择类别 留出训练集
        data_loader = PromptDataLoader(
            dataset=datasets,
            tokenizer=tokenizer,
            template=promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
        )


        promptModel.eval()
        with torch.no_grad():
            for batch in data_loader:
                logits = promptModel(batch)
                preds = torch.argmax(logits, dim=-1)
                #print(classes[preds])
                result_matrix[count_line][preds] += 1
        count_line += 1
        #print(result_matrix)
    return result_matrix

def get_classes():
    #print("1")
    f_get_classes = open(f"./data/{args.dataset}/classes.txt","r")
    class_lines = f_get_classes.readlines()
    remove_class_lines = []
    for item in class_lines:
        remove_class_lines.append(item.replace("\n",""))
    #print(remove_class_lines)
    return remove_class_lines
    #拿类名


import math
# 计算两点之间的距离
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

#################此处为对字典排序取出前x个  {"label1":"概率","label2":"概率"}
def dict_sort(d):
    x = args.top_sentences
    d.items()
    L = list(d.items())
    L.sort(key=lambda x: x[1], reverse=False)
    return dict(L[0:x])

def nearest_embedding(embedding_list,centre_result,clustering_result):
    distance_dict = [{} for _ in range(args.k)]
    count = 0
    for item in clustering_result:
        # item 是聚类结果
        distance = eucliDist(embedding_list[count],centre_result[item])
        #print(distance)
        distance_dict[item][count] = distance
        count += 1
    #print(distance_dict)
    top_nearest = []
    for item in distance_dict:
        sorted_item = dict_sort(item)
        top_nearest.append(sorted_item)
        #print(sorted_item)
    return top_nearest

# [{1:1},{},{},{}]
def sort_nearest_sentences(top_nearest,line_list):
    result_layer_1_nearest = [[] for _ in range(args.k)]
    count = 0
    for item in top_nearest:
        for key,value in item.items():
            #print(key)
            #print(value)
            result_layer_1_nearest[count].append([count,line_list[key]])
        count += 1
    return result_layer_1_nearest

from sentence_transformers import InputExample, losses, evaluation
from torch.utils.data import DataLoader

#train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
def read_fine_tuning_data(csv_name):
    data_read = open(f"./scripts/fine_tune/{args.dataset}/{csv_name}.csv","r")
    data_list = data_read.readlines()
    train_set = []
    count = 0
    labels = ["World","Sports","Business","Science"]
    #labels = ["business","entertainment","health","sci_tech","sport","us","world"]
    for item in data_list:
        count += 1
        line_raw = item.split(",")
        text_a = line_raw[1]
        text_b = line_raw[2]
        label = line_raw[0]

        #if count%2==0:
        #    train_set.append(InputExample(texts=[text_a, text_b], label=float(label)))
        #else:
        train_set.append(InputExample(texts=[text_a + text_b,"This news topic is about " + labels[int(label)-1]], label=1.0))
        #train_set.append(InputExample(texts=[text_a,"This news topic is about " + labels[int(label)-1]], label=1.0))

        for index in range(args.k):
            if index != int(label)-1:
                train_set.append(
                    InputExample(texts=[text_a + text_b, "This news topic is about " + labels[index]],label=0.0))
                    #InputExample(texts=[text_a, "This news topic is about " + labels[index]],label=0.0))

    data_read.close()
    return train_set

def read_fine_tuning_dev_data(csv_name):
    data_read = open(f"./scripts/fine_tune/{args.dataset}/{csv_name}.csv","r")
    data_list = data_read.readlines()
    train_set = []
    count = 0
    labels = ["World", "Sports", "Business", "Science"]
    #labels = ["business","entertainment","health","sci_tech","sport","us","world"]
    for item in data_list:
        count += 1
        line_raw = item.split(",")
        text_a = line_raw[1]
        text_b = line_raw[2]
        label = line_raw[0]

        #if count%2==0:
        #    train_set.append([label,text_a,text_b])
        #else:
        train_set.append([1.0,"This news topic is about " + labels[int(float(label)-1)],text_a+text_b])
        #train_set.append([1.0,"This news topic is about " + labels[int(float(label)-1)],text_a])

        for index in range(args.k):
            if index != int(float(label)-1):
                train_set.append([0.0, "This news topic is about " + labels[index], text_a + text_b])
                #train_set.append([0.0, "This news topic is about " + labels[index], text_a])

    data_read.close()
    return train_set

def fine_tuning(sentence_model):
    train_set = read_fine_tuning_data("train")

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(sentence_model)

    # Define your evaluation examples
    # sentences1 = Ko_list[train_size:]
    # sentences2 = Cn_list[train_size:]
    # sentences1.extend(list(shuffle_Ko_list[train_size:]))
    # sentences2.extend(list(shuffle_Cn_list[train_size:]))
    # scores = [1.0] * eval_size + [0.0] * eval_size
    # evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    dev_set = read_fine_tuning_dev_data("dev")
    sentences1 = [i[1] for i in dev_set]
    sentences2 = [i[2] for i in dev_set]
    scores = [float(i[0]) for i in dev_set]
    # Tune the model
    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    sentence_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=args.fine_tune_epoch, warmup_steps=100,
                       evaluator=evaluator, output_path=f'./custom_model/{args.dataset}')
import math

def NMI(A, B):
    # 样本点数
    total = len(A)  # 数据及的个数
    A_ids = set(A)  # 创建一个无序不重复的集合，保存A簇的个数
    B_ids = set(B)  # 保存B簇的个数
    # 互信息计算
    MI = 0
    eps = 1.4e-45  # 防止出现log 0的情况
    for idA in A_ids:  # 对于A中的每一个簇
        for idB in B_ids:  # 对于B中的每一个簇
            idAOccur = np.where(A == idA)  # np.where(condition)，则输出满足条件 (即非0) 元素的坐标
            idBOccur = np.where(B == idB)  # 一个簇的元素所对应的位置
            idABOccur = np.intersect1d(idAOccur, idBOccur)  # 找到A/B位置的交集
            px = 1.0 * len(idAOccur[0])/total
            py = 1.0 * len(idBOccur[0])/total
            pxy = 1.0 * len(idABOccur)/total
            MI = MI + pxy * math.log(pxy/(px * py) + eps, 2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat


if __name__ == "__main__":
    lunci = 0
    sentence_model = SentenceTransformer('./model/all-MiniLM-L6-v2')

    fileHandler = open(f"./data/{args.dataset}/test.csv", "r",encoding="utf-8")
    fileWriter = open(f"./scripts/TextClassification/{args.dataset}/test.csv",  "w",encoding="utf-8")
    fileWriter.close()

    #print("now start fine tuning")
    if args.re_custom_model:
        fine_tuning(sentence_model)
    if args.use_custom_model:
        sentence_model = SentenceTransformer(f'./custom_model/{args.dataset}')
    #exit()
    fileWriter = open(f"./scripts/TextClassification/{args.dataset}/test.csv", "a")
    count, line_list, label_list = read_data(fileHandler , fileWriter)
    fileWriter.close()
    fileHandler.close()
    ##start
    embedding_list = sentence_model.encode(line_list)

    if args.umap:
        reducer = umap.UMAP(random_state=42)
        embedding_list = reducer.fit_transform(embedding_list)

    result_layer = k_means(embedding_list, args.k)

    fig, ax = plt.subplots()
    ax.scatter(embedding_list[:, 1], embedding_list[:, 0], c=label_list)
    plt.savefig(f"./img/{args.dataset}/raw.png")

    #print(result_layer[1])

    fig, ax = plt.subplots()
    ax.scatter(embedding_list[:, 1], embedding_list[:, 0], c=result_layer[1])
    plt.savefig(f"./img/{args.dataset}/deal.png")

    print("accuracy_score:"+str(sm.accuracy_score(label_list,result_layer[1])))
    print(label_list)
    print(result_layer[1])
    print("hungarian_cluster_acc:"+str(hungarian_cluster_acc(np.array(label_list,dtype=int),np.array(result_layer[1],dtype=int))))
    #print("f1_score:"+str(sm.f1_score(label_list,result_layer[1])))
    print("silhouette_score:"+str(sm.silhouette_score(embedding_list,result_layer[1])))
    #print(result_layer[1])
    print("sklearn_NMI:"+str(sm.normalized_mutual_info_score(label_list,result_layer[1])))
    print("sklearn_ARI:"+str(sm.adjusted_rand_score(label_list,result_layer[1])))
    #print("NMI:"+str(NMI(np.ndarray(embedding_list),np.ndarray(result_layer[1]))))

    '''
    while True:
        print("=========================================:" + str(lunci))
        lunci += 1

        fileWriter = open(f"./scripts/TextClassification/{args.dataset}/test.csv", "a")
        count, line_list, label_list = read_data(fileHandler, args.batch, fileWriter)
        fileWriter.close()
        ##exit
        if count == 0:
            break

        ##start
        embedding_list = sentence_model.encode(line_list)

        if args.umap:
            reducer = umap.UMAP(random_state=42)
            embedding_list = reducer.fit_transform(embedding_list)

        #print(embedding_list)
        #第一轮用k means
        if lunci == 1:
            #result_layer_1 = k_means(embedding_list,args.k)

            fig, ax = plt.subplots()
            ax.scatter(embedding_list[:, 1], embedding_list[:, 0], c=label_list)
            plt.savefig(f"./img/{args.dataset}/cluster.png")
            print("1 layer")
            #exit()
            #print(result_layer_1[0]) # clustering center
            #print(result_layer_1[1]) # clustering result
            #break
            #top_nearest = nearest_embedding(embedding_list,result_layer_1[0],result_layer_1[1])

            #sorted_nearest_sentences = sort_nearest_sentences(top_nearest, line_list)
            #sorted_sentences = sort_sentences(result_layer_1[1],line_list)
            #save_cluster_sentence(sorted_sentences, f"./scripts/cluster/{args.dataset}/train_demo.csv")
            # 用 prompt zero-shot 分类 提供训练集
            #save_cluster_sentence(sorted_nearest_sentences, f"./scripts/cluster/{args.dataset}/train.csv")

            #result_matrix = prompt_classification_unsupervised(sorted_nearest_sentences)

            #print(result_matrix)

            #max_dimension1 = np.amax(result_matrix, axis=0)

            #np.amax_index
            #max_dimension2 = np.amax(result_matrix, axis=1)
            #print(max_dimension1)
            #print(result_matrix)
            #demo_ = AgnewsProcessor().get_train_examples("./scripts/cluster/agnews/")
            #print(demo_)

        else:
            fig, ax = plt.subplots()

            ax.scatter(embedding_list[:, 1], embedding_list[:, 0], c=label_list)

            plt.savefig(f"./img/{args.dataset}/cluster"+str(lunci)+".png")
            #print("1 layer")
            print("n layer")
            # 此处直接用prompt跑few-shot 做评价 NMI

        ###清空
        fileWriter = open(f'./scripts/TextClassification/{args.dataset}/test.csv', 'w', encoding='utf-8')
        fileWriter.close()
    
    fileHandler.close()
    '''