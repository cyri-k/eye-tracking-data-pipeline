import os
import glob
from pathlib import Path
from typing import Tuple, List
from collections import defaultdict
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

country_meta = {
    '0.Adult_control_group': {'label': 'Control Group', 'code': 'Ctrl'},
    '1.Austria':     {'label': 'Österreich', 'code': 'AT'},
    '2.Germany':     {'label': 'Deutschland', 'code': 'DE'},
    '3.Switzerland': {'label': 'Schweiz', 'code': 'CH'}
}
institution_meta = {
    'Primary_school':       {'label': 'Grundschule', 'code': 'GS'},
    'Kindergarten':         {'label': 'KiGa', 'code': 'KG'},
    'Dialekt (Konstanz)': {'label': 'Konstanz', 'code': 'KN'},
    'Standard (Marburg)': {'label': 'Marburg', 'code': 'MR'}
}
def generate_metatable(input_dir: str) -> pd.DataFrame:
    """
    Generates a metadata DataFrame by scanning a directory for .hdf5 experimental files.

    This function iterates through .hdf5 files within the specified input directory,
    extracts relevant metadata from their file paths and names, and organizes it
    into a pandas DataFrame. It handles different file naming conventions and
    identifies multiple runs for the same subject.

    Args:
        input_dir (str): The path to the directory containing the .hdf5 experiment files.

    Returns:
        pd.DataFrame: A DataFrame where each row represents an .hdf5 file and contains
                      extracted metadata such as subject code, country, institution,
                      experiment version, file size, date, time, and flags for
                      control group and multiple attempts.

    Raises:
        ValueError: If no .hdf5 files are found in the specified input directory,
                    or if a file name does not conform to expected formats.
    """
    meta_table = pd.DataFrame()
    trial_files = glob.glob(input_dir + '/**/*.hdf5', recursive=True)

    if not trial_files:
        raise ValueError(f"No .hdf5 files found in: {input_dir}")
    
    meta_dict = defaultdict(list)
    for exp_file_path in trial_files:
        # print(f'{country_code}_{institution_code}_{exp_version}')
        
        if 'Adult_control_group' in exp_file_path:
            if_control_group = True
            country, institution, edit_label, exp_version, hdf5_file_name = exp_file_path.split('\\')[-5:] 
            relative_path = os.path.join(country, institution, edit_label, exp_version)
        else:
            if_control_group = False
            country, institution, exp_version, hdf5_file_name=exp_file_path.split('\\')[-4:]
            edit_label = None
            relative_path = os.path.join(country, institution, exp_version)
        
        file_size = os.path.getsize(exp_file_path)
        split_name = hdf5_file_name.split('_')
        
        if len(split_name) == 6: # Older subject code format
            institution_code = "None"
            subject_code = split_name[0]
        elif len(split_name) == 8: # Updated subject code format
            institution_code = split_name[1]
            subject_code = split_name[2]
        else:
            raise ValueError(f"Non-standard file name: {hdf5_file_name}")
        
        date = split_name[-2]
        time = split_name[-1][:-5]                    
    
        output_file_name = f"{subject_code}_{{}}_{country_meta[country]['code']}_{institution_meta[institution]['code']}_{exp_version}.csv" # Later inserted with Run
        
        # Record subjects with multiple attempts under the same key
        meta_dict[output_file_name].append({
            "out": output_file_name, 
            "Subject": int(subject_code),
            "Country": country_meta[country]['label'],
            "Institution": institution_meta[institution]['label'],
            "Version": exp_version,
            "Run": 1,                   # Updated later
            "conversionSuccess": None,  # Updated later
            "hasMultiple": 0,           # Updated later
            "FileSize": file_size,
            "Date": date,
            "Time": time,
            "in": hdf5_file_name,
            "missingCount": None,       # Updated later
            "missingPercent": None,     # Updated later
            "numberOfTrials": None,     # Updated later
            "Session": 1 if if_control_group else None,# Updated later     
            "institutionCode": institution_code,
            "path": relative_path,
            "ifControlGroup": if_control_group,
            "ifSubjectHasPairAB": None, # Updated later
            "conversionError": None     # Updated later 
        })
        
    # Handle subjects with multiple attempts
    for file_list in meta_dict.values():
        if len(file_list) > 1:
            def convert_time(time_str):
                return datetime.strptime(time_str, '%Hh%M.%S.%f')
            for run_label, file_dict in enumerate(sorted(file_list, key=lambda x: convert_time(x["Time"])), 1):
                file_dict["hasMultiple"] = 1
                file_dict["Run"] = run_label
                file_dict["out"] = file_dict["out"].format(run_label) # Insert run_label into {} placeholder
        else: 
            file_list[0]["out"] = file_list[0]["out"].format(1) # Insert 1 into {} placeholder if there is only one run
        meta_table = pd.concat([meta_table, pd.DataFrame(file_list)],ignore_index=True)
    return meta_table