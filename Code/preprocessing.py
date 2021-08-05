from IPython.display import display, Markdown,Latex,HTML
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import _pickle as cPickle


def load(stage='raw',reapply=False,input_path='../Data',input_name='oasis3'):
  
    input_file = f'{input_path}/processed/{stage}_{input_name}.pkl'
  
    if Path(input_file).exists() or not reapply:
        data = load_pickle(input_file)
    else:
        pipeline = {'raw':   download_data,
                    'clean': clean_data,
                    'bmi':   process_bmi }
    
        if stage in pipeline:
            data = pipeline[stage](input_file)
        else:
            raise Exception(f"Sorry, {stage} is not a valid pipeline stage")
            
    return data  


def download_data(input_file):
    data = {}
    for d in ["ADRCClinicalData",'FreeSurfers','SubDemos',"subjects"]:
        url = f'https://raw.githubusercontent.com/esoreq/Real_site/master/data/{d}.csv'
        data[d] = pd.read_csv(url)
    save_pickle(input_file,data)  
    return data 

def missing_profile(x):
    d = {} 
    d['notnull'] = x.notnull().sum()
    d['isnull'] = x.isnull().sum()
    d['%missing'] = d['isnull']/x.shape[0]
    return pd.Series(d, index=d.keys())


def drop_missing_columns(df,thr):
    _df = df.apply(missing_profile).T
    columns_2_drop = _df[_df['%missing']>thr]
    if not columns_2_drop.empty:
        df = df.drop(columns=columns_2_drop.index)
    
    return df,columns_2_drop 
  

def clean_data(output_file,thr = 0.9):
    data = load('raw')
    dropped = {}
    for k in data.keys():
        data[k],dropped[k] = drop_missing_columns(data[k],thr)
    save_pickle(output_file,data)
    return data,dropped


def days_since_entry(df):
    df['days_since_entry'] = df['pid'].apply(lambda x: int(x.split('_')[-1][1:]))
    return df

def process_bmi(output_file):
    # load cleaned data
    df = load('clean')["ADRCClinicalData"]  # extract only task-relevant columns
    df = df.iloc[:,[0,1,3,18,19]] 
    # rename columns for simplicity 
    df.columns = ['pid','subj','age_at_entry','height','weight']
    # extract days_since_entry
    df = days_since_entry(df)
    df['age'] = df['days_since_entry']/365.2425 + df['age_at_entry']
    # transform from Imperial to metric 
    df['height'] = df['height']*0.0254 # transfrom from inch to meters
    df['weight'] = df['weight']*0.453592 # transfrom from pounds to kg
    # calculate bmi
    df['bmi'] = df['weight']/(df['height']**2)
    # categorize bmi
    bins=[0,18.5,25,30,60]
    labels=['Underweight','Normal','Overweight','Obese']
    df['bmi_cat'] = pd.cut(df['bmi'],bins,labels=labels)
    save_pickle(output_file,df)
    return df 


def get_parent(filename):
    return Path(filename).absolute().parent
  
def save_pickle(output_name,data):
    if not Path(output_name).exists():
        Path(get_parent(output_name)).mkdir(parents=True, exist_ok=True)
    output_pickle = open(output_name, "wb")
    cPickle.dump(data, output_pickle)
    output_pickle.close()
       
def load_pickle(input_name):
    input_pickle = open(input_name, 'rb')
    data = cPickle.load(input_pickle)
    input_pickle.close()
    return data    