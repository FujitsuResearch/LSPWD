import numpy as np
from scipy.optimize import linear_sum_assignment

def temporal_iou(pred_span, gt_span):
    """
    Calculate 1D Intersection over Union (IoU) between two temporal spans.
    Args:
        pred_span (tuple/list): Predicted temporal span (start, end)
        gt_span (tuple/list): Ground truth temporal span (start, end)
    Returns:
        float: IoU score between 0 and 1
    """
    pred_start, pred_end = pred_span
    gt_start, gt_end = gt_span
    
    # Ensure valid spans
    if pred_end < pred_start or gt_end < gt_start:
        raise ValueError("End time cannot be before start time")
        
    # Calculate intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    if intersection_end <= intersection_start:
        return 0.0
        
    intersection = intersection_end - intersection_start
    
    # Calculate union
    pred_duration = pred_end - pred_start
    gt_duration = gt_end - gt_start
    union = pred_duration + gt_duration - intersection
    
    # Calculate IoU
    iou = intersection / union
    
    return float(iou)

def match_temporal_iou(preds, gts):
    """
    Find optimal matching between predicted and ground truth temporal spans using Hungarian algorithm.
    
    Args:
        preds (list): List of predicted temporal spans, each span is [start, end]
        gts (list): List of ground truth temporal spans, each span is [start, end]
    
    Returns:
        tuple: (matched_indices, total_iou)
            - matched_indices: List of (pred_idx, gt_idx) pairs
            - total_iou: Sum of IoUs for the matched pairs
    """
    if not preds or not gts:
        return [], 0.0
    
    # Calculate cost matrix (negative IoU since Hungarian algorithm minimizes cost)
    cost_matrix = np.zeros((len(preds), len(gts)))
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            cost_matrix[i, j] = -temporal_iou(pred, gt)  # Negative since we want to maximize IoU
    
    # Apply Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Get matched pairs and total IoU
    matched_pairs = list(zip(pred_indices, gt_indices))
    total_iou = -cost_matrix[pred_indices, gt_indices].sum()  # Convert back to positive
    avg_iou = total_iou /  len(gts)
    
    return matched_pairs, avg_iou
'''
# Example usage:
if __name__ == "__main__":
    # Example predictions and ground truths
    predictions = [[10, 20], [25, 35], [40, 50], [50, 55]]
    ground_truths = [[15, 25], [30, 40], [45, 55]]
    
    # Find optimal matching
    matches, avg_iou = match_temporal_iou(predictions, ground_truths)
    
    print("Matched pairs (pred_idx, gt_idx):", matches)
    print("Avg IoU:", avg_iou)
    
    # Print individual IoUs for matched pairs
    print("\nIndividual IoUs:")
    for pred_idx, gt_idx in matches:
        iou = temporal_iou(predictions[pred_idx], ground_truths[gt_idx])
        print(f"Pred {pred_idx} - GT {gt_idx}: {iou:.3f}")
'''


def find_difference_range(s1, s2):
    # Ignore first and last chars by slicing [1:-1]
    s1_mid = s1[1:-1]
    s2_mid = s2[1:-1]
    
    n = len(s1_mid)
    if n != len(s2_mid):
        return None  # Strings of different lengths
        
    # Find start of difference
    start = 0
    while start < n and s1_mid[start] == s2_mid[start]:
        start += 1
        
    # Find end of difference (going backwards)
    end = n - 1
    while end >= start and s1_mid[end] == s2_mid[end]:
        end -= 1
        
    # Adjust indices to account for ignored first character
    return [start + 1, end + 1] if start <= end else None

'''
# Test with your example
s1 = "GIBJBIGCHEHCGIBFAD-"
s2 = "GIBJBIGCHED----FADG"
result = find_difference_range(s1, s2)
print(f"Different substrings: '{s1[result[0]:result[1]+1]}' and '{s2[result[0]:result[1]+1]}'")
'''



def get_overlapping_substring(s1, s2, best_offset, max_matches):
    len1 = len(s1)
    len2 = len(s2)
    start_index_s1 = -1
    start_index_s2 = -1

    for i in range(len1):
        j = i - best_offset
        if 0 <= j < len2 and s1[i] == s2[j]:
            start_index_s1 = i
            start_index_s2 = j
            break # Find the first index of match

    if start_index_s1 != -1:
        return s1[start_index_s1 : start_index_s1 + max_matches]
    else:
        return ""

'''
string1 = 'JBHKHBJGCEID'
string2 = 'BJGCEIDIALFKCGJ'
best_offset, max_matches = align_strings(string1, string2)
overlapping_part = get_overlapping_substring(string1, string2, best_offset, max_matches)
print("String 1:", string1)
print("String 2:", string2)
print("\nOverlapping part:", overlapping_part)
'''


def find_difference_range(s1, s2):
    # Ignore first and last chars by slicing [1:-1]
    s1_mid = s1[1:-1]
    s2_mid = s2[1:-1]
    
    n = len(s1_mid)
    if n != len(s2_mid):
        return None  # Strings of different lengths
        
    # Find start of difference
    start = 0
    while start < n and s1_mid[start] == s2_mid[start]:
        start += 1
        
    # Find end of difference (going backwards)
    end = n - 1
    while end >= start and s1_mid[end] == s2_mid[end]:
        end -= 1
        
    # Adjust indices to account for ignored first character
    return [start + 1, end + 1] if start <= end else None
'''
# Test with your example
s1 = "GIBJBIGCHEHCGIBFAD-"
s2 = "GIBJBIGCHED----FADG"
result = find_difference_range(s1, s2)
print(f"Different substrings: '{s1[result[0]:result[1]+1]}' and '{s2[result[0]:result[1]+1]}'")
print(f"Range index: {result}")
'''