import pandas as pd
import numpy as np
# import matplotlib as plt
import os
from tqdm import tqdm
from collections import defaultdict
import shutil, sys
import datetime
import pickle
dest_path = "..\my_dataset_final"
walking_data = []
onTable_data = []
onLaptop_data = []
all_data = defaultdict(None)
labels_str = ['labelWalking', 'labelJogging' ,'labelStairs' ,'labelSitting' ,'labelStanding' ,'labelTyping' ,'labelBrushingTeeth' ,'labelEatingSoup' ,'labelEatingChips' ,
'labelEatingPasta' ,'labelDrinkingfromCup' ,'labelEatingSandwich' ,'labelKicking' ,'labelPlayingCatch','labelDribbling' ,'labelWriting','labelClapping ' ,'labelFoldingClothes' ]
print(len(labels_str))
print(os.getcwd())
columns_acc_p = ['user_id','label','timestamp','acc_x_phone','acc_y_phone','acc_z_phone']
columns_acc_w = ['user_id','label','timestamp','acc_x_watch','acc_y_watch','acc_z_watch']
columns_gyr_p = ['user_id','label','timestamp','gyr_x_phone','gyr_y_phone','gyr_z_phone']
columns_gyr_w = ['user_id','label','timestamp','gyr_x_watch','gyr_y_watch','gyr_z_watch']

devices = ['phone','watch']
users = range(1600,1650)

data_acc_path_p = os.path.join(os.getcwd(),'..','..','raw_data','WISDM','phone','accel') 
data_gyr_path_p = os.path.join(os.getcwd(),'..','..','raw_data','WISDM','phone','gyro') 
data_acc_path_w = os.path.join(os.getcwd(),'..','..','raw_data','WISDM','watch','accel') 
data_gyr_path_w = os.path.join(os.getcwd(),'..','..','raw_data','WISDM','watch','gyro') 
for user in users:
    data_acc_p = pd.read_csv(os.path.join(data_acc_path_p,'data_'+str(user)+'_accel_phone.txt') ,names= columns_acc_p )
    data_gyr_p = pd.read_csv(os.path.join(data_gyr_path_p,'data_'+str(user)+'_gyro_phone.txt'),names= columns_gyr_p)
    data_acc_w = pd.read_csv(os.path.join(data_acc_path_w,'data_'+str(user)+'_accel_watch.txt'),names= columns_acc_w)
    data_gyr_w = pd.read_csv(os.path.join(data_gyr_path_w,'data_'+str(user)+'_gyro_watch.txt'),names= columns_gyr_w)
    


    #clean unwanted characters
    data_acc_p['acc_z_phone'] = data_acc_p['acc_z_phone'].map(lambda x: float(str(x).strip(';')))
    data_gyr_p['gyr_z_phone'] = data_gyr_p['gyr_z_phone'].map(lambda x: float(str(x).strip(';')))
    data_acc_w['acc_z_watch'] = data_acc_w['acc_z_watch'].map(lambda x: float(str(x).strip(';')))
    data_gyr_w['gyr_z_watch'] = data_gyr_w['gyr_z_watch'].map(lambda x: float(str(x).strip(';')))

    #makedummies for labels
    dummies = pd.get_dummies(data_acc_p['label'])
    data_acc_p = pd.concat([data_acc_p, dummies], axis=1)
    dummies = pd.get_dummies(data_gyr_p['label'])
    data_gyr_p = pd.concat([data_gyr_p, dummies], axis=1)
    dummies = pd.get_dummies(data_acc_w['label'])
    data_acc_w = pd.concat([data_acc_w, dummies], axis=1)
    dummies = pd.get_dummies(data_gyr_w['label'])
    data_gyr_w = pd.concat([data_gyr_w, dummies], axis=1)


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
    data_acc_p['timestamp'] = data_acc_p['timestamp'].map(lambda x:  handle_time(x) ) 
    data_gyr_p['timestamp'] = data_gyr_p['timestamp'].map(lambda x:  handle_time(x) ) 
    data_acc_w['timestamp'] = data_acc_w['timestamp'].map(lambda x:  handle_time(x) ) 
    data_gyr_w['timestamp'] = data_gyr_w['timestamp'].map(lambda x:  handle_time(x) ) 
    
    
    #merge devices and sensors
    data = pd.concat([data_acc_p, data_gyr_p,data_acc_w,data_gyr_w], axis=0)
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