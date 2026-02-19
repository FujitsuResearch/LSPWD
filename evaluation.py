import numpy as np
import json
import glob
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.periodic_detection_helper import *
from utils.eval import *

def evaluate():
    output_dir = 'outputs'
    output_files = glob.glob(os.path.join(output_dir, '*.json'))
    
    if not output_files:
        print("No output files found in 'outputs/' directory.")
        return

    MAPEs = []
    avg_IoUs = []    
    MAEs = []
    IoUs = []

    for output_file in output_files:
        with open(output_file, 'r') as f:
            estimation = json.load(f)
        
        test_file_path = estimation['test_file_path']
        print(f"Evaluating: {test_file_path}")
        
        try:
            with open(test_file_path+'/GT.json', 'r') as file:
                GT = json.load(file)
        except FileNotFoundError:
            print(f"GT file not found for: {test_file_path}")
            continue

        ### Task 1 Evaluation
        period_num = estimation['task_1']['period_num']
        period_boundaries = estimation['task_1']['period_boundaries']
        
        # Convert keys back to int if they are strings (JSON keys are always strings)
        # But match_temporal_iou expects a list of boundaries [start, end]
        # In estimation.py, period_boundaries is a dict {p_id: [start, end]}
        # We need to convert it to list of values
        
        period_boundaries_list = list(period_boundaries.values())

        MAPE = abs(GT['task_1']['p_num'] - period_num) / GT['task_1']['p_num']
        matches, avg_iou = match_temporal_iou(period_boundaries_list, GT['task_1']['boundaries'])
    
        #print('num of periods: {}'.format(period_num))
        #print('Task 1: MAPE: {}'.format(MAPE)) 
        #print('Task 1: avg IoU: {}'.format(avg_iou)) 
        
        MAPEs.append(MAPE)
        avg_IoUs.append(avg_iou)   

        ### Task 2 Evaluation
        est_remained_ratio = estimation['task_2']['est_remained_ratio']
        
        # We need to recalculate GT_remained_ratio here as it wasn't saved
        # Or we can rely on GT being available
        observed_idx = int(np.sum(GT['task_2']['seg_idx'][:GT['task_2']['clip_idx']]))
        remained_idx = np.sum(GT['task_2']['seg_idx'][GT['task_2']['clip_idx']:])
        GT_remained_ratio = remained_idx / np.sum(GT['task_2']['seg_idx'])

        MAE = abs(est_remained_ratio - GT_remained_ratio)

        #print('Task 2: MAE: {}'.format(MAE)) 
        MAEs.append(MAE)

        ### Task 3 Evaluation
        est_abnormal_range = estimation['task_3']['est_abnormal_range']
        
        IoU = temporal_iou(est_abnormal_range, GT['task_3'])

        #print('Task 3: IoU: {}'.format(IoU))
        IoUs.append(IoU)

    print('MAPE: {}'.format(np.mean(MAPEs)))
    print('avg IoU: {}'.format(np.mean(avg_IoUs)))
    print('MAE: {}'.format(np.mean(np.nan_to_num(MAEs, nan=1.0))))
    print('IoU: {}'.format(np.mean(IoUs)))

if __name__ == "__main__":
    evaluate()
