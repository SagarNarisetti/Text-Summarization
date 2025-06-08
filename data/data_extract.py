import os
import pandas as pd
import zipfile
data_directory = 'data'
data_zip = 'news_summary.zip'
extracted_data = 'extracted_data'
zip_file_path = os.getcwd()

with open(os.path.join(zip_file_path, data_directory,data_zip), 'rb') as file_obj:
    print('datareaddonefrom', os.path.join(zip_file_path, data_directory))
    with zipfile.ZipFile(file_obj) as zip_ref:
        zip_ref.extractall(os.path.join(zip_file_path, data_directory,extracted_data))
        print("Files extracted successfully!")