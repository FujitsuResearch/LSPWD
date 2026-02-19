import numpy as np
import json
import glob
import sys
import os


# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.periodic_detection_helper import *
from utils.eval import *

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def estimate():
    token_num = 10
    file_path_list = glob.glob('data/*/*')
    
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for test_file_path in file_path_list:
        print(f"Processing: {test_file_path}")
        test_file_path = test_file_path.replace('\\','/')
        
        try:
            with open(test_file_path+'/period_traj.json', 'r') as file:
                trajectories = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {test_file_path}/period_traj.json")
            continue

        # Task 1 Estimation
        traj_task_1 = np.vstack(list(trajectories.values())[:3])
        traj_task_1 = traj_task_1.reshape(traj_task_1.shape[0],-1)
        cluster_labels, hard_token, soft_token, centroids = spatiotemporal_clustering(traj_task_1, token_num)
        sequence = number_to_alpha(cluster_labels)
        num_frames = len(sequence) 

        window_sizes, magnitudes = dominant_fourier_frequency_2d(soft_token, lbound=10, ubound=max(len(soft_token.T), len(soft_token))//2)

        ### optimize win size
        scores = []
        for win in window_sizes[:10]: # select top 10 window sizes
            temporal_buffer = int(win*0.2)
            periods = []
            for i in range(num_frames//win):
                clip = sequence[max(0, win*i-temporal_buffer):min(num_frames, win*(i+1)+temporal_buffer )]
                periods.append(clip)
            
            compressed_periods = []
            for p in periods:
                compressed_periods.append(fuse_adjacent(p))
            score = calculate_similarity_score(compressed_periods)
            scores.append(score)
    
        if len(scores) > 0:
            win = window_sizes[np.argmax(scores)]
        else:
            win = 10 # Default fallback if no window sizes found
            
        # print('selected_win:{}'.format(win))
        temporal_buffer = int(win*0.2)
        periods = []
        for i in range(num_frames//win):
            clip = sequence[max(0, win*i-temporal_buffer):min(num_frames, win*(i+1)+temporal_buffer )]
            periods.append(clip)

        compressed_periods = []
        for p in periods:
            compressed_periods.append(fuse_adjacent(p))

        if len(compressed_periods) >= 3:
            aligned_sequences = msa(compressed_periods[:3])
        else:
            aligned_sequences = compressed_periods # Fallback if not enough periods

        # Clean up aligned sequences
        while aligned_sequences and '-' in [x[-1] for x in aligned_sequences]:
            i = find_dash_end_index(aligned_sequences)
            if i!=0:
                aligned_sequences = [s[:i] for s in aligned_sequences]
            else:
                break

        i = find_longest_repeated_ends(aligned_sequences)
        if i!=0:
            aligned_sequences = [s[:-i] for s in aligned_sequences]

        workflow_str = summarize_strings(aligned_sequences)

        # Task 1 Output Preparation
        workflow_str_len = len(workflow_str)
        # print(workflow_str_len)

        workflow = [[] for _ in range(workflow_str_len)]
        for seq in aligned_sequences:
            pointer = 0
            Flag = False
            
            pos_skip_sign = seq.find('-')
            if pos_skip_sign==-1: pos_skip_sign = workflow_str_len //2
            pos_skip_sign = min(pos_skip_sign, workflow_str.find('_'))
            pos_skip_sign = max(pos_skip_sign, 1)

            for i in range(len(seq)):
                l = seq[i]
                if pointer==workflow_str_len:
                    break
                if seq[i:i+pos_skip_sign] == workflow_str[:pos_skip_sign]:
                    Flag = True
                if Flag:
                    workflow[pointer].append(l.replace("-", "_")+'{:02}'.format(pointer))
                    pointer += 1

        try:
            workflow_multi_paths = np.stack([''.join([y[0] for i, y in enumerate(x)]) for x in np.stack(workflow).T])
        except:
            workflow_multi_paths = []
    
        seg_labels = {}
        seg_ind = -1
        transcript_pointer = -1
        workflow_str_len = len(workflow_str)
        workflow_section_len = {}
        for frame_number, l in enumerate(sequence):
            if len(workflow_str) > 0 and l==workflow_str[0] and workflow_str[transcript_pointer]==workflow_str[-1]:
                if seg_ind == -1 or len(seg_labels[seg_ind]) > 0.4 * win:
                    transcript_pointer = 0
                    seg_ind += 1
                    seg_labels[seg_ind] = {}
                    workflow_section_len[seg_ind] = {}
                    workflow_section_len[seg_ind][transcript_pointer] = 0
            if transcript_pointer==-1: continue
            if transcript_pointer < workflow_str_len-1:
                if l == workflow_str[transcript_pointer+1]:
                    transcript_pointer += 1
                    workflow_section_len[seg_ind][transcript_pointer] = 0
            if transcript_pointer < workflow_str_len-1:
                if workflow_str[transcript_pointer+1]=='_':
                    transcript_pointer += 1
                    workflow_section_len[seg_ind][transcript_pointer] = 0

            if transcript_pointer == workflow_str_len-1 and workflow_section_len[seg_ind][transcript_pointer]>1 and l != workflow_str[transcript_pointer]:
                continue

            seg_labels[seg_ind][frame_number] = l
            workflow_section_len[seg_ind][transcript_pointer] +=1

        workflow_section_len = [v for k,v in workflow_section_len.items() if len(v)>workflow_str_len*0.3]
        workflow_section_len_array = []
        for idx in range(len(workflow_section_len)):
            workflow_section_len_array.append(list(workflow_section_len[idx].values()))

        if len(workflow_section_len_array)>0:
            sublist_max_len = max(len(sublist) for sublist in workflow_section_len_array)
            workflow_section_len_array = [sublist for sublist in workflow_section_len_array if len(sublist)==sublist_max_len]
            workflow_section_len_array = np.stack(workflow_section_len_array)
            workflow_section_len = np.median(workflow_section_len_array,0)
        else:
            workflow_section_len = np.zeros(workflow_str_len)

        ### Task 1 Result
        period_num = len([x for x in seg_labels.values() if len(x)>0.5*win])
        if period_num>0:
            period_boundaries = {}
            for p_id, (k,v) in enumerate(seg_labels.items()):
                frame_list = np.sort(list(v.keys()))
                # Convert numpy int64 to python int for JSON serialization
                period_boundaries[p_id] = [int(frame_list[0]), int(frame_list[-1])]
                if p_id > 0: period_boundaries[p_id-1][1] =  int(frame_list[0]-1)
        else:
            period_num = len(traj_task_1)//win
            period_boundaries = {i:[int((i-1)*win), int(i*win)] for i in range(1,period_num+1)}


        ### Task 2 Estimation
        traj_task_2 = np.vstack(list(trajectories.values())[3])
        hard_tokens_task_2 = [np.sum(np.abs((centroids - x)),1).argmin() for x in traj_task_2]
        hard_tokens_task_2 = number_to_alpha(hard_tokens_task_2)
        
        # Note: Task 2 estimation use GT['task_2']['seg_idx'] to know the observed part (input) but not the remained part (output).
        # read `GT.json` to get `clip_idx` and `seg_idx` to define the input for Task 2.
        
        with open(test_file_path+'/GT.json', 'r') as file:
            GT = json.load(file)

        observed_idx = int(np.sum(GT['task_2']['seg_idx'][:GT['task_2']['clip_idx']]))

        observed_portion = hard_tokens_task_2[:observed_idx]

        compressed_obs = fuse_adjacent(observed_portion)
        best_offset, max_matches = align_strings(compressed_obs, workflow_str)
        overlapping_part = get_overlapping_substring(compressed_obs, workflow_str, best_offset, max_matches)
        
        est_remained_ratio = 0.0
        if len(workflow_section_len) < workflow_str.find(overlapping_part) + len(overlapping_part):
            if len(workflow_str) > 0:
                est_remained_ratio = len(compressed_obs)/len(workflow_str)
        else:
            denom = np.sum(workflow_section_len)
            if denom > 0:
                est_remained_ratio = np.sum(workflow_section_len[workflow_str.find(overlapping_part) + len(overlapping_part):]) / denom

        ### Task 3 Estimation
        traj_task_3 = np.vstack(list(trajectories.values())[4])
        hard_tokens_task_3 = [np.sum(np.abs((centroids - x)),1).argmin() for x in traj_task_3]
        hard_tokens_task_3 = number_to_alpha(hard_tokens_task_3)

        compressed_task_3 = fuse_adjacent(hard_tokens_task_3)

        abnormal_range = [0, len(compressed_task_3)]
        for workflow_path in workflow_multi_paths:
            aligned_sequences = msa([workflow_path, compressed_task_3])
            temp_abnormal_range = find_difference_range(aligned_sequences[0], aligned_sequences[1])
            if temp_abnormal_range is None: continue
            if (temp_abnormal_range[1] - temp_abnormal_range[0]) < (abnormal_range[1] - abnormal_range[0]):
                abnormal_range = temp_abnormal_range
        
        est_abnormal_range = [float(np.sum(workflow_section_len[:abnormal_range[0]]-1)), float(np.sum(workflow_section_len[:abnormal_range[1]+1]))]

        # Save results
        output_data = {
            'test_file_path': test_file_path,
            'task_1': {
                'period_num': period_num,
                'period_boundaries': period_boundaries
            },
            'task_2': {
                'est_remained_ratio': float(est_remained_ratio)
            },
            'task_3': {
                'est_abnormal_range': est_abnormal_range
            }
        }

        # Create a unique filename for the output
        # Assuming test_file_path is like 'data/Category/Sample'
        # We can replace '/' with '_'
        output_filename = test_file_path.replace('data/', '').replace('/', '_') + '.json'
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as outfile:
            json.dump(output_data, outfile, indent=4, cls=NumpyEncoder)
        
        print(f"Saved estimation to {output_path}")

if __name__ == "__main__":
    estimate()
