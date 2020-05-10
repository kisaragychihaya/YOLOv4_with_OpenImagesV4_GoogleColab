import os
import shutil
import pandas as pd
import argparse
import glob2
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def get_name_to_id(obj_file_path):
    f = open(obj_file_path,'r')
    name_to_id = {}
    class_id = 0
    for i in f:
        i = i.split('\n')[0]
        name_to_id[i] = class_id
        class_id += 1
    return name_to_id

def get_code_to_name(name_to_id,
                     class_descriptions_path= '/content/OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csv'):
    class_descriptions_df = pd.read_csv(class_descriptions_path,names=['classes_coded','classes_names'])
    code_to_names = dict(zip(class_descriptions_df['classes_coded'],class_descriptions_df['classes_names']))
    return code_to_names

def get_list_files_name(images_path='/content/OIDv4_ToolKit/OID/Dataset/test'):
    files_name = list()
    for classes_imanges in os.listdir(images_path):
        files_name.extend(os.listdir(os.path.join(images_path,classes_imanges)))
    return files_name

def format_labels_data(list_files_name,code_to_names,name_to_id,out_path,
                       annotations_file_path='/content/OIDv4_ToolKit/OID/csv_folder/test-annotations-bbox.csv' ):
                       
    annotations_df_raw = pd.read_csv(annotations_file_path)
    images_id= [file_name.split('.')[0] for file_name in list_files_name 
                if file_name.split('.')[-1] in ["png", "jpeg", "jpg"]]

    code_available = [i for i in code_to_names if code_to_names[i] in name_to_id]
    annotations_df = annotations_df_raw.loc[annotations_df_raw['ImageID'].isin(images_id) &
                                            annotations_df_raw['LabelName'].isin(code_available)]
    annotations_df['width'] = annotations_df['XMax'] - annotations_df['XMin']
    annotations_df['height'] = annotations_df['YMax'] - annotations_df['YMin']
    annotations_df['X'] = (annotations_df['XMax'] + annotations_df['XMin'])/2
    annotations_df['Y'] = (annotations_df['YMax'] + annotations_df['YMin'])/2
    # return annotations_df
    if not os.path.isdir(out_path):
        os.mkdir(out_path) 
    for _, row in annotations_df.iterrows():
        with open('{}/{}.txt'.format(out_path,row['ImageID']),'a') as f:
            f.write(' '.join([str(i) for i in [name_to_id[code_to_names[row['LabelName']]]
                                               ,row["X"], row['Y'], row['width'], row['height']]])+os.linesep)

def copy_data(data_path,folder_des):
    classes = os.listdir(data_path)

    for class_ in classes:
        labels_folder = os.path.join(data_path,class_)
        label_files = os.listdir(labels_folder)
        for label_file in label_files:
            if label_file.split('.')[-1] in ["png", "jpeg", "jpg"]:
                shutil.copyfile(os.path.join(labels_folder,label_file),
                                os.path.join(folder_des,label_file))

def preprocess_data(data_set_name='Train',des_path='custom_data' ,
                    ojd_file_path='obj_name.txt', path_dataset='OIDv4_ToolKit/OID'):
    
    if not os.path.isdir(des_path):
        os.mkdir(des_path)  

    data_set_name = data_set_name.lower()

    name_to_id = get_name_to_id(ojd_file_path)

    class_descriptions_path = os.path.join(path_dataset,'csv_folder/class-descriptions-boxable.csv')
    code_to_names = get_code_to_name(name_to_id,class_descriptions_path)
    
    images_path = os.path.join(path_dataset,f'Dataset/{data_set_name}')
    files_name = get_list_files_name(images_path)

    annotations_file_path = os.path.join(path_dataset,f'csv_folder/{data_set_name}-annotations-bbox.csv')
    format_labels_data(files_name,code_to_names,name_to_id,des_path,annotations_file_path)

    copy_data(images_path,des_path)

def split_data(data_path='custom_data', yolo_path='', test_size=0.2):

    all_files = []
    for ext in ["*.png", "*.jpeg", "*.jpg"]:
        images = glob2.glob(os.path.join(data_path, ext))
        all_files += images
    # print(len(all_files))
    rand_idx = np.random.randint(0, len(all_files), int(test_size*len(all_files)))

    # Create train.txt
    with open(os.path.join(yolo_path,"train.txt"), "w") as f:
        for idx in np.arange(len(all_files)):
            if idx not in rand_idx:
                f.write(all_files[idx]+'\n')

    # Create valid.txt
    with open(os.path.join(yolo_path,"valid.txt"), "w") as f:
        for idx in np.arange(len(all_files)):
            if idx in rand_idx:
                f.write(all_files[idx]+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set_name', type=str,
                         default='Train')
    parser.add_argument('--des_path', type=str,
                         default='custom_data')
    parser.add_argument('--ojd_file_path', type=str,
                         default='obj_name.txt')
    parser.add_argument('--path_dataset', type=str,
                         default='OIDv4_ToolKit/OID')

    parser.add_argument('--yolo_path', type=str,
                         default='')
    parser.add_argument('--test_size', type=float,
                         default=0.2)

    args = parser.parse_args()
    data_set_name = args.data_set_name
    des_path = args.des_path
    ojd_file_path = args.ojd_file_path
    path_dataset = args.path_dataset

    preprocess_data(data_set_name,des_path,
                    ojd_file_path,path_dataset)

    yolo_path = args.yolo_path
    test_size = args.test_size

    split_data(des_path,
             yolo_path, test_size)




    

