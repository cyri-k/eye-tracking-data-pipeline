import os
from typing import Tuple, List

import h5py
import pandas as pd
import numpy as np

cols_to_remove =['left_gaze_z','left_angle_x','left_angle_y', 'left_raw_x', 'left_raw_y',\
    'left_pupil_measure1_type','left_pupil_measure2','left_pupil_measure2_type',\
    'left_ppd_x', 'left_ppd_y','left_velocity_x', 'left_velocity_y', 'left_velocity_xy',\
    'right_gaze_z','right_angle_x','right_angle_y', 'right_raw_x', 'right_raw_y',\
    'right_pupil_measure1_type','right_pupil_measure2','right_pupil_measure2_type',\
    'right_ppd_x', 'right_ppd_y','right_velocity_x', 'right_velocity_y', 'right_velocity_xy',\
                'device_time', 'logged_time',\
    'confidence_interval', 'delay', 'filter_id',\
    'left_eye_cam_x', 'left_eye_cam_y', 'left_eye_cam_z',\
    'right_eye_cam_x', 'right_eye_cam_y', 'right_eye_cam_z']

def AOI_mapper(row):
    if row['region']=="":
        return ""
    else:
        tl,tr,br,bl=row.posPerm.strip('[]').split(' ')
        if row['region']=="tl":
            return int(tl)
        elif row['region']=="tr":
            return int(tr)
        elif row['region']=="br":
            return int(br)
        elif row['region']=="bl":
            return int(bl)
        else:
            print("Am here")
            return ""

def clean_pupil(row):
    left = row['left_pupil_measure1']
    right = row['right_pupil_measure1']
    
    if pd.notna(left) and pd.notna(right) and left > 0 and right > 0:
        return (left + right) / 2

    elif pd.notna(left) and left > 0:
        return left
    elif pd.notna(right) and right > 0:
        return right
    else:
        return np.nan  # both missing or invalid



def eye_tracking_hdf5_to_df(data_file_path: str) -> dict:
    file_name = data_file_path.split('\\')[-1]
    with h5py.File(data_file_path, 'r') as hdf: 
        try:
            # Handle meta data
            session_meta_data = hdf['data_collection']['session_meta_data']
            recorded_session = int(session_meta_data['code'][0])
            
            trial_file_csv = data_file_path.replace('.hdf5','.csv')
            trial_meta_df = pd.read_csv(trial_file_csv,skiprows=range(1,7),usecols=range(0,6))
            trial_meta_df = trial_meta_df.dropna()

            if trial_meta_df.empty:
                raise ValueError(f"Meta file is empty")
            
            # Handle MessageEvent
            events_data = hdf['/data_collection/events/experiment/MessageEvent']
            events_df=pd.DataFrame(np.array(events_data))
            if events_df.empty:
                raise ValueError(f"MessageEvent is empty")

            events_df.text=events_df.text.str.decode('utf-8')
            

            if events_df.loc[events_df.text=='trackerTest2022_2_5'].shape[0]!=1:
                #print(events_df.loc[events_df.text=='trackerTest2022_2_5'])
                raise ValueError(events_df.loc[events_df.text=='trackerTest2022_2_5'])
            
            events_df=events_df.drop(['device_time','logged_time','confidence_interval','delay','msg_offset','category'],axis=1)

            start_index=events_df.loc[events_df.text=="trackerTest2022_2_5"].index.tolist()[0]
            start_time=events_df.iloc[start_index]['time']
            
            end_index=events_df.iloc[start_index:].index.tolist()[-1]
            end_time=events_df.iloc[end_index]['time']
            
            # Handle Eye Tracking Data
            eye_tracking_data = hdf['/data_collection/events/eyetracker/BinocularEyeSampleEvent']
            eye_tracking_df=pd.DataFrame(np.array(eye_tracking_data))
            
            eye_tracking_df=eye_tracking_df.drop(cols_to_remove,axis=1)
            eye_tracking_df=eye_tracking_df.loc[eye_tracking_df['time'].between(start_time,end_time)]

            if eye_tracking_df.empty:
                raise ValueError(f"BinocularEyeSampleEvent is empty")
            
            
            # eye_tracking_df=eye_tracking_df.loc[eye_tracking_df.status==0] # Drops trackless data 
            eye_tracking_df['gaze_x']=eye_tracking_df[['left_gaze_x','right_gaze_x']].mean(axis=1)
            eye_tracking_df['gaze_y']=eye_tracking_df[['left_gaze_y','right_gaze_y']].mean(axis=1)

            eye_tracking_df['pupil_size'] = eye_tracking_df.apply(clean_pupil, axis=1)   
            
            eye_tracking_df.loc[eye_tracking_df.gaze_x.between(-0.75,-0.25) & eye_tracking_df.gaze_y.between(0.0,0.5,inclusive='neither'),"region"]="tl"
            eye_tracking_df.loc[eye_tracking_df.gaze_x.between(0.25,0.75) & eye_tracking_df.gaze_y.between(0.0,0.5,inclusive='neither'),"region"]="tr"
            eye_tracking_df.loc[eye_tracking_df.gaze_x.between(0.25,0.75) & eye_tracking_df.gaze_y.between(-0.5,0.0,inclusive='neither'),"region"]="br"
            eye_tracking_df.loc[eye_tracking_df.gaze_x.between(-0.75,-0.25) & eye_tracking_df.gaze_y.between(-0.5,0.0,inclusive='neither'),"region"]="bl"
            eye_tracking_df=eye_tracking_df.drop(['left_gaze_x', 'left_gaze_y', 'left_pupil_measure1', 'right_gaze_x','right_gaze_y', 'right_pupil_measure1', 'status'],axis=1)
            eye_tracking_df.region=eye_tracking_df.region.fillna("")
            
            # Process meta data
            trial_meta_df = pd.read_csv(trial_file_csv,skiprows=range(1,7),usecols=range(0,6))  # Trial metadata
            trial_meta_df=trial_meta_df.reset_index()
            trial_meta_df['trial']=trial_meta_df.index+1
            trial_meta_df=trial_meta_df.drop(trial_meta_df.tail(1).index)
            trial_meta_df=trial_meta_df.drop('index',axis=1)
            trial_meta_df=trial_meta_df.melt(id_vars=['trial','sound','condition'],value_vars=['target_image','phonological_image','semantic_image','unrelated_image'],value_name='image',var_name='AOI')
            trial_meta_df = trial_meta_df.sort_values('trial').reset_index(drop=True)
            
        
            tstart_events = events_df[events_df['text'].str.startswith('tStart', na=False)].copy()
            tstart_events['trial'] = tstart_events['text'].str.extract(r'tStart\s+(\d+)').astype(int)+1
            tstart_events['end_time'] = events_df.loc[events_df['text'].str.startswith('tEnd', na=False),'time'].tolist()
            
            if len(events_df.loc[events_df['text'].str.startswith('Target:', na=False)].text.str.split(': ',expand=True)[3].tolist())==20:
                tstart_events['posPerm'] = events_df.loc[events_df['text'].str.startswith('Target:', na=False)].text.str.split(': ',expand=True)[3].tolist()
            else: # TODO: Save experiment with fewer trials anyway
                raise ValueError(f'Not 20: {data_file_path}')
            tstart_events=tstart_events.rename(columns={'time': 'start_time'})
            
            tstart_events = tstart_events.sort_values('start_time').reset_index(drop=True)    
            eye_tracking_df = eye_tracking_df.sort_values('time').reset_index(drop=True)

            
            trial_df=pd.merge_asof(eye_tracking_df, tstart_events[['start_time','trial','end_time','posPerm']], left_on='time',right_on='start_time')
            trial_df = trial_df[
                (trial_df['time'] >= trial_df['start_time']) &
                (trial_df['time'] < trial_df['end_time'])
            ]
            
            trial_df['AOI']=trial_df.apply(AOI_mapper,axis=1)
            trial_df['trial'] = trial_df['trial'].astype(int)
            trial_df=trial_df.replace({'AOI': {0: 'target_image', 1: 'phonological_image', 2:'semantic_image',3:'unrelated_image'}})
            trial_df=trial_df.merge(trial_meta_df[['trial','sound','condition']].drop_duplicates(),on=['trial'])
            trial_df=trial_df.merge(trial_meta_df,on=['trial','sound','condition','AOI'],how='left')
            
            trial_df=trial_df.drop(['experiment_id','session_id','device_id','event_id','type'],axis=1)
            trial_df['TimeFromTrialOnset']=trial_df.time-trial_df.start_time

            trial_df['image'] = trial_df['image'].fillna(value='')
            trial_df['trackloss']=False
            trial_df.loc[trial_df.gaze_y.isna()==True,'trackloss']=True

            return {
                "conversionSuccess": 1,
                "session": recorded_session,
                "errorMessage": None,
                "df": trial_df
            }
        
        except Exception as e:
            # raise e
            return {
                "conversionSuccess": 0,
                "session": None,
                "errorMessage": f"Error: {str(e)}",
                "df": None
            }
            
class IDTFixationSaccadeClassifier:
    def __init__(self, threshold: float = 0.03, win_len: int = 12):
        self.threshold = threshold
        self.win_len = win_len

    @staticmethod
    def _calc_min_max_dispersion(x_range: np.array, y_range: np.array) -> int:
        dx = np.max(x_range) - np.min(x_range)
        dy = np.max(y_range) - np.min(y_range)
        return dx + dy

    def fit_predict(self, x: np.array, y: np.array = None) -> Tuple[List[int], List[int], List[int], List[int]]:

        if y is None:
            tmp = x.copy()
            x, y = tmp[:, 0], tmp[:, 1]

        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y shape does not match')

        fixations, saccades = [], []
        fixation_colors, saccades_colors = [], []

        win_beg, win_end = 0, min(len(x) - 1, self.win_len)

        fix_c, sacc_c = 0, 0

        while (win_beg < len(x)):
            if np.isnan(x[win_beg: win_end]).any() or np.isnan(y[win_beg: win_end]).any():
                win_beg += 1
                win_end = min(len(x), win_beg + self.win_len)
                continue

            dispersion = self._calc_min_max_dispersion(x[win_beg: win_end], y[win_beg: win_end])
            if dispersion < self.threshold:
                while win_end < len(x) - 1 and dispersion < self.threshold:
                    win_end += 1
                    dispersion = self._calc_min_max_dispersion(x[win_beg: win_end], y[win_beg: win_end])
                if win_end - win_beg > 10:
                    for i in range(win_beg, win_end):
                        fixations.append(i)
                        fixation_colors.append(fix_c)
                    fix_c += 1
                win_beg, win_end = win_end + 1, min(len(x), win_end + 1 + self.win_len)
            else:
                if len(saccades) > 0 and win_beg - 1 != saccades[-1]:
                    sacc_c += 1
                saccades.append(win_beg)
                saccades_colors.append(sacc_c)
                win_beg += 1

        return fixations, saccades, fixation_colors, saccades_colors
        
def aggregate_processed_data(output_dir: str, meta_table: pd.DataFrame) -> pd.DataFrame:
    meta_table = meta_table[meta_table.conversionSuccess == 1].set_index('out')
    data_tables = []
    for data_table_file in meta_table[meta_table.conversionSuccess == 1].index: # meta_table[(meta_table.missingPercent == 0) & (meta_table.Run == 1) & (meta_table.numberOfTrials == 20) & (meta_table.Country == 'Deutschland')].index:
        data_table = pd.read_csv(os.path.join(output_dir, f'{data_table_file}'), dtype={'fixation':'Int64', 'saccades':'Int64'})
        
        # Get normalized time
        for trial in data_table.trial.unique():
            trial_df = data_table[data_table.trial == trial] 
            
            onset_time = data_table.loc[data_table.trial == trial, 'time'].iloc[0]
            trial_times = data_table.loc[data_table.trial == trial, 'time']
            data_table.loc[data_table.trial == trial, 'timeFromOnsetMs'] = trial_times - onset_time - 3
            
        data_table.loc[:, ['in','Subject','Country','Institution','Version', 'Session','missingPercent','Run', 'numberOfTrials','ifControlGroup']] = meta_table.loc[data_table_file, ['in','Subject','Country','Institution','Version', 'Session','missingPercent','Run', 'numberOfTrials','ifControlGroup']].values
        data_table.loc[:, 'out'] = meta_table.loc[data_table_file,:].name

        data_tables.append(data_table)
    return pd.concat(data_tables, ignore_index=True)
    
if __name__ == "__main__":
    input_dir=r'C:\Users\Cyril\HESSENBOX\Eye-Tracking_LAVA (Jasmin Devi Nuscheler)\Data_from_different_participants'
    output_dir=r'C:\dev\grk-2700\eye_tracking_pipeline\tests\results'
    data_file_path = 'C:\\Users\\Cyril\\HESSENBOX\\Eye-Tracking_LAVA (Jasmin Devi Nuscheler)\\Data_from_different_participants\\2.Germany\\Primary_school\\A\\069_trackerTest2022_2_5_2024-06-13_08h05.39.112.hdf5'
    eye_tracking_hdf5_to_df(data_file_path)