import pandas as pd
import numpy as np
import arff

# import matplotlib as plt
import os
from tqdm import tqdm
from collections import defaultdict
import shutil, sys
import datetime
import pickle
dest_path = "..\my_dataset_final"

all_data = defaultdict(None)
labels_str = ['labelWalking', 'labelJogging' ,'labelStairs' ,'labelSitting' ,'labelStanding' ,'labelTyping' ,'labelBrushingTeeth' ,'labelEatingSoup' ,'labelEatingChips' ,
'labelEatingPasta' ,'labelDrinkingfromCup' ,'labelEatingSandwich' ,'labelKicking' ,'labelPlayingCatch','labelDribbling' ,'labelWriting','labelClapping ' ,'labelFoldingClothes' ]
print(len(labels_str))
print(os.getcwd())
columns_acc_p = ['label','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVR','YAVR','ZAVR','XPEAK','YPEAK','ZPEAK','XABSOLDEV','YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','XVAR','YVAR','ZVAR','XMFCC0','XMFCC1','XMFCC2','XMFCC3','XMFCC4','XMFCC5','XMFCC6','XMFCC7','XMFCC8','XMFCC9','XMFCC10','XMFCC11','XMFCC12','YMFCC0','YMFCC1','YMFCC2','YMFCC3','YMFCC4','YMFCC5','YMFCC6','YMFCC7','YMFCC8','YMFCC9','YMFCC10','YMFCC11','YMFCC12','ZMFCC0','ZMFCC1','ZMFCC2','ZMFCC3','ZMFCC4','ZMFCC5','ZMFCC6','ZMFCC7','ZMFCC8','ZMFCC9','ZMFCC10','ZMFCC11','ZMFCC12','XYCOS','XZCOS','YZCOS','XYCOR','XZCOR','YZCOR','RESULTANT','USER']
columns_acc_w = ['label','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVR','YAVR','ZAVR','XPEAK','YPEAK','ZPEAK','XABSOLDEV','YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','XVAR','YVAR','ZVAR','XMFCC0','XMFCC1','XMFCC2','XMFCC3','XMFCC4','XMFCC5','XMFCC6','XMFCC7','XMFCC8','XMFCC9','XMFCC10','XMFCC11','XMFCC12','YMFCC0','YMFCC1','YMFCC2','YMFCC3','YMFCC4','YMFCC5','YMFCC6','YMFCC7','YMFCC8','YMFCC9','YMFCC10','YMFCC11','YMFCC12','ZMFCC0','ZMFCC1','ZMFCC2','ZMFCC3','ZMFCC4','ZMFCC5','ZMFCC6','ZMFCC7','ZMFCC8','ZMFCC9','ZMFCC10','ZMFCC11','ZMFCC12','XYCOS','XZCOS','YZCOS','XYCOR','XZCOR','YZCOR','RESULTANT','USER']
columns_gyr_p =['label','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVR','YAVR','ZAVR','XPEAK','YPEAK','ZPEAK','XABSOLDEV','YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','XVAR','YVAR','ZVAR','XMFCC0','XMFCC1','XMFCC2','XMFCC3','XMFCC4','XMFCC5','XMFCC6','XMFCC7','XMFCC8','XMFCC9','XMFCC10','XMFCC11','XMFCC12','YMFCC0','YMFCC1','YMFCC2','YMFCC3','YMFCC4','YMFCC5','YMFCC6','YMFCC7','YMFCC8','YMFCC9','YMFCC10','YMFCC11','YMFCC12','ZMFCC0','ZMFCC1','ZMFCC2','ZMFCC3','ZMFCC4','ZMFCC5','ZMFCC6','ZMFCC7','ZMFCC8','ZMFCC9','ZMFCC10','ZMFCC11','ZMFCC12','XYCOS','XZCOS','YZCOS','XYCOR','XZCOR','YZCOR','RESULTANT','USER']
columns_gyr_w = ['label','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVR','YAVR','ZAVR','XPEAK','YPEAK','ZPEAK','XABSOLDEV','YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','XVAR','YVAR','ZVAR','XMFCC0','XMFCC1','XMFCC2','XMFCC3','XMFCC4','XMFCC5','XMFCC6','XMFCC7','XMFCC8','XMFCC9','XMFCC10','XMFCC11','XMFCC12','YMFCC0','YMFCC1','YMFCC2','YMFCC3','YMFCC4','YMFCC5','YMFCC6','YMFCC7','YMFCC8','YMFCC9','YMFCC10','YMFCC11','YMFCC12','ZMFCC0','ZMFCC1','ZMFCC2','ZMFCC3','ZMFCC4','ZMFCC5','ZMFCC6','ZMFCC7','ZMFCC8','ZMFCC9','ZMFCC10','ZMFCC11','ZMFCC12','XYCOS','XZCOS','YZCOS','XYCOR','XZCOR','YZCOR','RESULTANT','USER']
columns_acc_p =['label'] + [ x+'_acc_phone' for x in columns_acc_p[1:]]
columns_acc_w =['label'] + [ x+'_acc_watch' for x in columns_acc_w[1:]]
columns_gyr_p =['label'] + [ x+'_gyr_phone' for x in columns_gyr_p[1:]]
columns_gyr_w =['label'] + [ x+'_gyr_watch' for x in columns_gyr_w[1:]]
devices = ['phone','watch']

users = range(1600,1650)

data_acc_path_p = os.path.join(os.getcwd(),'..','..','raw_data','WISDM','arff_files','phone','accel') 
data_gyr_path_p = os.path.join(os.getcwd(),'..','..','raw_data','WISDM','arff_files','phone','gyro') 
data_acc_path_w = os.path.join(os.getcwd(),'..','..','raw_data','WISDM','arff_files','watch','accel') 
data_gyr_path_w = os.path.join(os.getcwd(),'..','..','raw_data','WISDM','arff_files','watch','gyro') 
for user in users:

    # data = arff.loadarff('yeast-train.arff')
    # df = pd.DataFrame(data[0])
    # path = os.path.join(data_acc_path_p,'data_'+str(user)+'_accel_phone.arff')
    # print(path)
    # data = pd.DataFrame(list(arff.load(path)))
    # # data = list(data)
    # # data = arff.loadarff(path)
    # print(data)
    # exit()
    data_acc_p = pd.DataFrame(list(arff.load(os.path.join(data_acc_path_p,'data_'+str(user)+'_accel_phone.arff'))) ,columns= columns_acc_p )
    data_gyr_p = pd.DataFrame(list(arff.load(os.path.join(data_gyr_path_p,'data_'+str(user)+'_gyro_phone.arff'))),columns= columns_gyr_p)
    data_acc_w = pd.DataFrame(list(arff.load(os.path.join(data_acc_path_w,'data_'+str(user)+'_accel_watch.arff'))),columns= columns_acc_w)
    data_gyr_w = pd.DataFrame(list(arff.load(os.path.join(data_gyr_path_w,'data_'+str(user)+'_gyro_watch.arff'))),columns= columns_gyr_w)
    
    # print(data_acc_p)
    # exit()
    #clean unwanted characters
    # data_acc_p['acc_z_phone'] = data_acc_p['acc_z_phone'].map(lambda x: float(str(x).strip(';')))
    # data_gyr_p['gyr_z_phone'] = data_gyr_p['gyr_z_phone'].map(lambda x: float(str(x).strip(';')))
    # data_acc_w['acc_z_watch'] = data_acc_w['acc_z_watch'].map(lambda x: float(str(x).strip(';')))
    # data_gyr_w['gyr_z_watch'] = data_gyr_w['gyr_z_watch'].map(lambda x: float(str(x).strip(';')))

    #makedummies for labels
    dummies = pd.get_dummies(data_acc_p['label'])
    data_acc_p = pd.concat([data_acc_p, dummies], axis=1)
    dummies = pd.get_dummies(data_gyr_p['label'])
    data_gyr_p = pd.concat([data_gyr_p, dummies], axis=1)
    dummies = pd.get_dummies(data_acc_w['label'])
    data_acc_w = pd.concat([data_acc_w, dummies], axis=1)
    dummies = pd.get_dummies(data_gyr_w['label'])
    data_gyr_w = pd.concat([data_gyr_w, dummies], axis=1)
    
    data_acc_p = data_acc_p.drop(columns=['label'])
    data_acc_w = data_acc_w.drop(columns=['label'])
    data_gyr_p = data_gyr_p.drop(columns=['label'])
    data_gyr_w = data_gyr_w.drop(columns=['label'])
    # print(data_acc_p)
    # exit()

    ##rename columns
    # data_acc_p.columns = ['user_id','label','timestamp','acc_x_phone','acc_y_phone','acc_z_phone'] +labels_str
    # data_gyr_p.columns = ['user_id','label','timestamp','gyr_x_phone','gyr_y_phone','gyr_z_phone'] +labels_str
    # data_acc_w.columns = ['user_id','label','timestamp','acc_x_watch','acc_y_watch','acc_z_watch'] +labels_str
    # data_gyr_w.columns = ['user_id','label','timestamp','gyr_x_watch','gyr_y_watch','gyr_z_watch'] +labels_str
    # print(data.columns)
    # exit()
    def handle_time(timestamp):
        # print(timestamp)
        div = 100000000
        dt = datetime.datetime.fromtimestamp(int(timestamp)    // div)

        # print(dt)
        # exit()
        # s = dt.strftime('%Y-%m-%d %H:%M:%S')
        # s += '.' + str(int(int(timestamp) % div)).zfill(9)
        return dt

    #changing timestamp to date time format
    # data_acc_p['timestamp'] = data_acc_p['timestamp'].map(lambda x:  handle_time(x) ) 
    # data_gyr_p['timestamp'] = data_gyr_p['timestamp'].map(lambda x:  handle_time(x) ) 
    # data_acc_w['timestamp'] = data_acc_w['timestamp'].map(lambda x:  handle_time(x) ) 
    # data_gyr_w['timestamp'] = data_gyr_w['timestamp'].map(lambda x:  handle_time(x) ) 
    print(len(data_acc_p),len(data_gyr_p),len(data_acc_w),len(data_gyr_w))
    # exit()
    #merge devices and sensors
    data = pd.concat([data_acc_p, data_gyr_p,data_acc_w,data_gyr_w], axis=1)
    print(data.head(-10))
    print(data.isna().sum())
    exit()
    # print(data['timestamp'])
    ##aggregate data
    # data = data.groupby([ 'timestamp']).agg({'score': 'max'........}).reset_index()
    data.index = data['timestamp'] 
    data =data.resample('T').max()
    print(data.isna().sum())
    #save the csv for each user 
    if os.path.exists(dest_path+'/'+str(user)):
        pass
    else:
        os.mkdir(dest_path+'/'+str(user))
    data.to_csv(dest_path+'\\'+str(user)+'\\'+'all_data.csv', index=False)
    exit()