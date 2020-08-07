#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:55:44 2020

@author: leiliang
"""
#from features import transform_480,model_path,feature_length,get_feat
#import torch
#import torchvision

from SODmodule import preSOD
from features import get_feat
import copy
import sys
#from tqdm import tqdm
import numpy as np
#import warnings
import joblib
import csv
import codecs
import shutil
import os
import faiss
import time

def indextoID(name_list, result_name_list, predicted):
    dic=[]
    for i in range(len(predicted)):
        lis=[]
        lis.append(result_name_list[i].split(".")[0])
        for j in range(len(predicted[-1])):
            lis.append(name_list[predicted[i][j]].split(".")[0])
        dic.append(lis)
    return dic

def score(result_name_list, dict_result, var_path):
    recalltmp=[]
    precisiontmp=[]
    var_name_list=[]
    with open(var_path, 'r',encoding='utf-8-sig') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
                var_name_list.append(row)
        readFile.close()
    for k in range(len(dict_result)):
        recall=0
        precision=0
        val_rows=dict_result[k]
        for j in range(len(val_rows[1:])):
            for i in range(len(var_name_list[k])-1):
                if var_name_list[k][i+1]==val_rows[j+1]:
                    recall += 1
                    precision += recall/float(j+1)
        recalltmp.append(recall/(len(var_name_list[k])-1))
        if recall!=0:
            precisiontmp.append(precision/recall)
        else:
            precisiontmp.append(0)
    return recalltmp,precisiontmp

def testScore(valfeature_namlist, predict, var_path):
    Acc, Score = score(valfeature_namlist, predict, var_path)
    print("==>P> {}=>R> {}".format(np.mean(Score),np.mean(Acc)))

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
            file_csv = codecs.open(file_name,'w','utf-8')
            writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            for data in datas:
                writer.writerow(data)
            print("save result to %s success"%file_name)
            
def resultIDcsv(predict):
    resultdict=copy.deepcopy(predict)
    for i in range(len(predict)):
        for j in range(len(predict[0])):
            if j==0:
                resultdict[i][j]='<Test Image ID '+predict[i][j]+'#'+str(i+1)+'>'
            else:
                resultdict[i][j]='<Training Image ID '+predict[i][j]+'#'+str(i+1)+'_'+str(j)+'>'
    return resultdict

def faisssearch(test_image_path,result_path, mode='cpu',batch_size=100):
#    val_image_dir='/media/leiliang/HDDA/PF-500K/BP500K-master/val/val'
    start=time.time()
    feat_path='./feature/'
    var_path='./2020_val.csv'
    top_num=7
    val_image_dir=test_image_path
    model_name='seresnet152'
    model_name1='densenet201'
    weight={}
    weight[model_name]=10
    weight[model_name1]=8.95
    item = 'Grmac'
    featuretmp=joblib.load(feat_path+"/{}_feat_{}.pkl".format(item,model_name))
    feature_namlist=featuretmp["name"]
    feature1=featuretmp[item]
    valfeature_namlist,valfeaturetmp=get_feat(image_dir=val_image_dir,model_name=model_name,batch_size=batch_size,mode=mode)
    valfeature1=valfeaturetmp[item].numpy().astype("float16")

    item1 = 'Mac'
    featuretmp=joblib.load(feat_path+"/{}_feat_{}.pkl".format(item1,model_name1))
    _,valfeaturetmp=get_feat(image_dir=val_image_dir,model_name=model_name1,batch_size=batch_size,mode=mode)
    
    feature=np.column_stack((feature1*weight[model_name],featuretmp[item1]* weight[model_name1]))
    valfeature=np.column_stack((valfeature1*weight[model_name],valfeaturetmp[item1].numpy().astype("float16")* weight[model_name1]))
    
    index_feat = faiss.IndexFlatIP(feature.shape[1]) #IndexFlat    #IndexFlatIP  #IndexFlatL2
    index_feat.add(feature.astype("float32"))
    sims,Index = index_feat.search(valfeature.astype("float32"),top_num)
    print("search  time :%4f"%(time.time()-start))
#            print("total time :%4f"%(time.time()-start))
    predictID=Index.tolist()
    predict=indextoID(feature_namlist,valfeature_namlist,predictID)
    testScore(valfeature_namlist, predict, var_path)
    resultdict=resultIDcsv(predict)
    data_write_csv(result_path, resultdict)

if __name__ == "__main__":
    # test_image_path=sys.argv[1]
    # result_path=sys.argv[2]
    test_image_path='/media/leiliang/HDDA/PF-500K/2020-Val-Image'
    result_path='./result/predictions.csv'
    start=time.time()
    shutil.rmtree('./preimage')  
    os.mkdir('./preimage')
    preSOD(image_dir=test_image_path,predi_dir='./preimage')
    print("process  time :%4f"%(time.time()-start))
    faisssearch('./preimage',result_path,mode='cuda',batch_size=20)



