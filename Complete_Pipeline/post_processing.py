"""
    Created by Jahnavi Malagavalli, Dec 2023

    Usage:  This code processes the predictions given out by main.py file
    python post_processing.py --path <<path to the file or track directory>>
"""

import pandas as pd
import json
from collections import Counter
import ast
import argparse
import os 
from datetime import datetime
import glob
import copy


# Define a function to get the top N key-value pairs based on values
def top_keys_values(dictionary, n, no_of_rows):
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

    sorted_items_n = {key:round(x / no_of_rows, 2) for key,x in sorted_items[:n]}

    return sorted_items_n

# Define a function to remove the third word from the first element of a string tuple to a dictionary
def tuple_str_to_dict_vmmr(tuple_str):
    # Evaluate the string as a literal tuple
    tuple_value = ast.literal_eval(tuple_str)
    dict_mmr = {}
    for tuple_ in tuple_value:
      # Extract the brand and value from the tuple
      brand = tuple_[0].rsplit(' ', 1)[0]
      value = tuple_[1]
      dict_mmr[brand]=value

    return dict_mmr

def tuple_str_to_dict_ft_mmr(tuple_str):
    # Evaluate the string as a literal tuple
    tuple_value = ast.literal_eval(tuple_str)
    dict_mmr = {}
    for tuple_ in tuple_value:
      # Extract the brand and value from the tuple
      brand = tuple_[0]
      value = tuple_[1]
      dict_mmr[brand]=value

    return dict_mmr

# Use apply to accumulate values for each key across dictionaries
def update_counters(dictionary, result_counter, count_per_key):
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            result_counter[key] += value
            count_per_key[key] += 1


def post_processing(df,file):

    output_list = []
    grouped_df = df.groupby('vehc_id', as_index=False)
    # Display the result
    for name,group in grouped_df:
        output = {'timestamp':'', 'filename':file,'vehc_id':'','freq_vehc_cls':'','freq_vehc_cls_conf':'',
    'avg_vehc_speed':'','freq_vehc_colr_1':'','freq_vehc_colr_2':'',
    'freq_mmr_1':'','freq_mmr_2':'','freq_lp_chrs':'','freq_lp_state_chrs':''}

        print(f"vehc_id: {name}")

        # Calculate the mode of the 'vehc_cls' column
        mode_vehc_cls = group['vehc_cls'].mode()[0]
        print("frequent class:",mode_vehc_cls)
        #The coming aggregations are all calculated based on the highest frequency vehicle type
        group = group[group['vehc_cls'] == mode_vehc_cls]

        # mean of the conf
        mean_vehc_conf = group['vehc_conf'].mean()
        print(f"vehicle : {mode_vehc_cls} confidence: {mean_vehc_conf}")

        # mean of the speed which in miles/hour (orig it in in mps)
        mean_vech_speed = round(group[group['vehc_cls'] == mode_vehc_cls]['vehc_speed'].mean()*2.23694,2)
        print(f"speed after converion:{mean_vech_speed:.2f}")

        # mode of the color_m1 and max of color_m2
        # Apply the aggregation function to the 'DictionaryColumn'
        no_of_rows = group.shape[0]
        # Use Counter and apply to sum values for each key across dictionaries
        result_counter = Counter()
        group['color_m2'].apply(lambda x: result_counter.update(x) if isinstance(x, dict) else None)
        result_dict = dict(result_counter)
        max_color_m2 = top_keys_values(result_dict, 3, no_of_rows)
        mode_color_m1 = group['color_m1'].mode().tolist()
        print(f"most probable vehc colors from color_m1:{mode_color_m1} and color_m2:{max_color_m2}")

        #output.update({'vehc_id'=name,'freq_vehc_cls':mode_vehc_cls,'freq_vehc_cls_conf':mean_vehc_conf,\
        #'avg_vehc_speed':mean_vech_speed,'freq_vehc_colr_1':mode_color_m1,'freq_vehc_colr_2':max_color_m2})

        #mode of the car make and model
        # Use Counter and apply to sum values for each key across dictionaries

        #Initialize counters
        result_counter = Counter()
        count_per_key = Counter()
        # Apply the function to update counters
        group['car_mmr'].apply(update_counters, args=(result_counter, count_per_key))

        # Calculate the average value for each key
        average_per_key = {key: round(result_counter[key] / count_per_key[key],2) for key in result_counter}
        max_car_mmr = top_keys_values(average_per_key, 3, 1)

        #Initialize counters
        result_counter = Counter()
        count_per_key = Counter()
        # Apply the function to update counters
        group['car_sf_v_mmr'].apply(update_counters, args=(result_counter, count_per_key))

        # Calculate the average value for each key
        average_per_key = {key: round(result_counter[key] / count_per_key[key],2) for key in result_counter}
        max_car_sf_v_mmr = top_keys_values(average_per_key, 3, 1)

        print(f"most probable vehc ft_mmr: {max_car_mmr} and \nsfv_mmr:{max_car_sf_v_mmr}")

        #licence plate error rate
        #There can be at most 8 charecters on the licence plate apart from the US State name
        #This is the LP prediction format: dsfadf;
        #Find the 10 most repeated charecters
        data_without_na_lp = [entry for entry in group["LicensePlate"] if entry != 'NA']

        combined_text_lp = ''.join(data_without_na_lp)

        data_without_na_state = [entry for entry in group["state"] if entry != 'NA']

        combined_text_state = ''.join(data_without_na_state)

        if combined_text_lp=="":
            print("No LP Found")
            top_10_chars_lp='NA'
        
        else:
            # Use Counter to count the occurrences of each character
            char_counts = Counter(combined_text_lp)

            # Get the top 10 most common characters
            top_10_chars_lp = char_counts.most_common(10)

            print("Top 10 most common characters in LP:",top_10_chars_lp)

        if combined_text_state=="":
            print("No state found on LP")
            top_10_chars_st = 'NA'
        else:
            # Use Counter to count the occurrences of each character
            char_counts = Counter(combined_text_state)

            # Get the top 10 most common characters
            top_10_chars_st = char_counts.most_common(10)

            print("Top 10 most common characters in state:",top_10_chars_st)
        output.update({'vehc_id':name,'freq_vehc_cls':mode_vehc_cls,'freq_vehc_cls_conf':mean_vehc_conf,
         'avg_vehc_speed':mean_vech_speed,'freq_vehc_colr_1':mode_color_m1,'freq_vehc_colr_2':max_color_m2,
         'freq_mmr_1':max_car_mmr,'freq_mmr_2':max_car_sf_v_mmr,'freq_lp_chrs':top_10_chars_lp,'freq_lp_state_chrs':top_10_chars_st})
        
        output_list.append(copy.deepcopy(output))
        

        #print(group)
        print()
    return output_list


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Check if a given path is a directory or a file.")
    parser.add_argument("--path", required=True, help="The path to be checked.")

    args = parser.parse_args()
    path_to_check = args.path
    print("path to check:",path_to_check)
    #For Directory
    if os.path.isdir(path_to_check):
        #files = os.listdir(path_to_check)
        files = glob.glob(path_to_check+'**/*.txt', recursive=True)
        print("files:",files)
        #files = [os.path.join(path_to_check, file) for file in files]

    #For file
    elif os.path.isfile(path_to_check):
        files = [path_to_check]

    else:
        print("something is wrong")

    output = []
    for file in files:
        # Read the text file into a list of dictionaries
        with open(file, 'r') as fl:
            data = [json.loads(line) for line in fl]

        # Convert the list of dictionaries to a Pandas DataFrame
        df = pd.DataFrame(data)

        # List of columns to convert to float
        columns_to_convert = ['vehc_conf', 'vehc_speed']
        # Convert selected columns from object to float
        df[columns_to_convert] = df[columns_to_convert].replace('', '0')
        df[columns_to_convert] = df[columns_to_convert].astype(float)

        # Use apply with lambda functions to create new columns
        df['color_m1'] = df['vehc_color'].apply(lambda x: x[0])
        df['color_m2'] = df['vehc_color'].apply(lambda x: x[1])
        df['state'] = df['LP_txt'].apply(lambda x: x.split(';')[0])
        df['LicensePlate'] = df['LP_txt'].apply(lambda x: 'NA' if x == 'NA' else x.split(';')[1])
        
        df['car_mmr'] = df['car_mmr'].apply(tuple_str_to_dict_ft_mmr)
        df['car_sf_v_mmr'] = df['car_sf_v_mmr'].apply(tuple_str_to_dict_vmmr)

        filename = os.path.basename(file)
        output_dict_list = post_processing(df,filename)
        output = output + output_dict_list

    '''
    if(len(files)==1):
        file_path = 'Results/preprocessed_outputs/'+files[0].rsplit('/', 1)[1].split('.')[0]+'.json'
        print("file_path:",file_path)
    else:
        file_path = 'Results/preprocessed_outputs/output.json'
    # Serialize the dictionary and write it to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(output, json_file,indent=2)

    print(f"Data has been stored in {file_path}")
    '''


    # Convert the additional data to a DataFrame
    df_additional = pd.DataFrame(output)

    # Add a new column 'timestamp' with the current timestamp
    df_additional['timestamp'] = datetime.now()

    # Specify the existing Excel file path
    excel_file_path = 'Complete_Pipeline/Results/preprocessed_outputs/output.xlsx'

    # Check if the Excel file exists
    if os.path.exists(excel_file_path):
        # Load the existing Excel file into a DataFrame
        df_existing = pd.read_excel(excel_file_path)

        # Concatenate the existing DataFrame with the additional data
        df_combined = pd.concat([df_existing, df_additional], ignore_index=True)

        # Write the combined DataFrame to the Excel file
        df_combined.to_excel(excel_file_path, index=False)

        print(f'The additional data with timestamp has been appended to {excel_file_path}')
    else:
        # If the file doesn't exist, write the additional data directly
        df_additional.to_excel(excel_file_path, index=False)
        print(f'The Excel file {excel_file_path} did not exist. A new file has been created.')