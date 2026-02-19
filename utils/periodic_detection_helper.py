import numpy as np
import math
from itertools import product

from tqdm import tqdm
from sklearn.cluster import KMeans, MeanShift
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

import copy



def smooth(period_labels, gap = 1):
    period_labels_copy = copy.deepcopy(period_labels)
    for i in range(gap,len(period_labels)-gap):
        counts = np.bincount(period_labels[i-gap:i+gap])
        value = np.argmax(counts)
        period_labels_copy[i] = value
    return period_labels_copy

def spatiotemporal_clustering(spatiotemporal_data: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clusters a 3D spatial trajectory (ignoring timestamps) using DBSCAN and tokenizes it.

    Args:
        spatiotemporal_data: An array of [frame, n_feats].
    Returns:
        A tuple containing:
          - cluster_labels: A numpy array of cluster labels.
          - hard_tokenized_trajectory: A numpy array representing the hard-encoded tokenized trajectory (cluster labels)
          - soft_tokenized_trajectory: A numpy array representing the soft-encoded tokenized trajectory (vector of normalized distance to all centroids)
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=20, n_init='auto')  
    cluster_labels = kmeans.fit_predict(spatiotemporal_data)

    cluster_labels = smooth(cluster_labels, gap = 1)
    # Hard-encoded tokenization for the trajectory using cluster labels.
    hard_tokenized_trajectory = cluster_labels

    # Get cluster centroids
    centroids = kmeans.cluster_centers_
    n_clusters = len(centroids)
    n_points = len(spatiotemporal_data)
    
    # Initialize array for soft tokenization
    soft_tokenized_trajectory = np.zeros((n_points, n_clusters))
    
    # Compute Euclidean distances to all centroids for each point
    for i in tqdm(range(n_points)):
        point = spatiotemporal_data[i]
        distances = np.array([np.linalg.norm(point - centroid) for centroid in centroids])
        #'''
        # Convert distances to similarities using exponential decay
        similarities = np.exp(-distances)
        
        # Normalize similarities to sum to 1
        soft_tokenized_trajectory[i] = similarities / np.sum(similarities) 

    return cluster_labels, hard_tokenized_trajectory.T, soft_tokenized_trajectory.T, centroids



def create_path(nodes):
    """
    Create a string representing a path from a list of node sets.
    
    Args:
    nodes (list): List of lists of node IDs. Each list of nodes is connected by
        an edge.
    
    Returns:
    str: String representing the path.
    """
    result = []
    # Initial edge
    result.append(f"{nodes[0][0]}->{nodes[1][0]}")
    
    current_idx = 1
    # Loop until all edges are processed
    while current_idx < len(nodes) - 1:
        sources = nodes[current_idx]
        targets = nodes[current_idx + 1]
        
        if len(sources) == 1 and len(targets) == 1:
            # One source, one target
            result.append(f"{sources[0]}->{targets[0]}")
        elif len(sources) == 1:
            # One source, multiple targets
            paths = [f"{sources[0]}->{target}" for target in targets]
            result.append(f"({', '.join(paths)})")
        elif len(targets) == 1:
            # Multiple sources, one target
            paths = [f"{source}->{targets[0]}" for source in sources]
            result.append(f"({', '.join(paths)})")
        else:
            # Multiple sources, multiple targets
            paths = []
            for i in range(len(sources)):
                paths.append(f"{sources[i]}->{targets[i]}")
            result.append(f"({', '.join(paths)})")
            
        current_idx += 1
    
    return ', '.join(result)

def summarize_strings(strings):
    """
    Summarize a list of strings by comparing characters at each position.
    
    If all strings have the same character at a position, that character is
    included in the result. If not, an underscore is included.
    
    Args:
    strings (list): List of strings to summarize
    
    Returns:
    str: Summary of the strings
    """
    if not strings:
        return ""
    
    # Get length of shortest string
    min_len = min(len(s) for s in strings)
    
    # Compare characters at each position
    result = []
    for i in range(min_len):
        chars = set(s[i] for s in strings)
        # If all strings have the same character at this position, use that
        # character. Otherwise, use an underscore.
        result.append("_" if len(chars) > 1 else strings[0][i])
       
    return "".join(result)

def find_dash_end_index(strings):
    """
    Find the index of the last dash in the strings that is
    immediately preceded by a letter.
    
    Args:
        strings (list): List of strings with same length
    
    Returns:
        int: Index of the last dash (if found) or -1
    """
    # Ensure all strings have same length
    if not all(len(s) == len(strings[0]) for s in strings):
        raise ValueError("Strings must be of equal length")
    
    # Iterate from the right
    for i in range(len(strings[0])-1, -1, -1):
        for s in strings:
            if s[i] == '-':
                # Check if previous char is letter
                if i > 0 and s[i-1].isalpha():
                    return i
            elif not s[i].isalpha():  # Skip if not dash or letter
                continue
    
    return -1  # No matching pattern found



def find_longest_repeated_ends(strings):
    """
    Find the longest prefix and suffix that are identical across all strings.

    Args:
    strings (list): List of strings to check.

    Returns:
    int: Length of the longest common prefix and suffix.
    """
    if not strings:
        return 0

    # Use the first string as a reference
    s = strings[0]
    n = len(s)
    max_len = 0

    # Iterate over possible prefix/suffix lengths
    for i in range(1, n // 2 + 1):
        prefix = s[:i]
        suffix = s[-i:]

        # Check if prefix equals suffix and appears in all strings
        if prefix == suffix and all(st.startswith(prefix) and st.endswith(suffix) for st in strings):
            max_len = i

    return max_len


def create_path(nodes):
    result = []
    result.append(f"{nodes[0][0]}->{nodes[1][0]}")
    
    current_idx = 0
    while current_idx < len(nodes) - 1:
        sources = nodes[current_idx]
        targets = nodes[current_idx + 1]
        
        if len(sources) == 1 and len(targets) == 1:
            result.append(f"{sources[0]}->{targets[0]}")
        elif len(sources) == 1:
            paths = [f"{sources[0]}->{target}" for target in targets]
            result.append(f"({', '.join(paths)})")
        elif len(targets) == 1:
            paths = [f"{source}->{targets[0]}" for source in sources]
            result.append(f"({', '.join(paths)})")
        else:
            paths = []
            for i in range(len(sources)):
                paths.append(f"{sources[i]}->{targets[i]}")
            result.append(f"({', '.join(paths)})")
            
        current_idx += 1
    
    return ', '.join(result)
    

def dominant_fourier_frequency_2d(matrix, lbound=10, ubound=1000):
    """
    Find the dominant Fourier frequencies of a 2D matrix within a window size range.

    Parameters
    ----------
    matrix : array-like
        The input 2D matrix
    lbound : int, optional
        The lower bound of the window size range. Default is 10.
    ubound : int, optional
        The upper bound of the window size range. Default is 1000.

    Returns
    -------L
    tuple
        period_condidates
        period_condidates_magnitudes
    """
    # Compute 2D FFT
    fourier = np.fft.fft2(matrix)
    
    # Get frequency components for temporal dimensions
    freq_x = np.fft.fftfreq(matrix.shape[1], 1)
    
    magnitudes_x = []
    window_sizes_x = []
    
    # Analyze horizontal frequencies (x-axis)
    for j, freq in enumerate(freq_x):
        if freq > 0:  # Only consider positive frequencies
            window_size = int(1 / freq)
            if window_size >= lbound and window_size < ubound:
                # Sum magnitudes across columns for this frequency
                mag = 0
                for i in range(matrix.shape[0]):
                    coef = fourier[i, j]
                    mag += math.sqrt(coef.real * coef.real + coef.imag * coef.imag)
                window_sizes_x.append(window_size)
                magnitudes_x.append(mag)

    '''
    # Handle cases where no valid frequencies are found        
    if len(magnitudes_x) == 0:
        warnings.warn(f"Could not extract valid horizontal frequencies. Using window_size={lbound}.")
        period_x = lbound
    else:
        period_x = window_sizes_x[np.argmax(magnitudes_x)]
    '''
    
    return np.array(window_sizes_x)[np.argsort(magnitudes_x)[::-1]], np.sort(magnitudes_x)[::-1]



def dominant_fourier_frequency_1d(time_series, lbound=10, ubound=1000):
    """
    Find the dominant Fourier frequency of the time series within a window size range.

    Parameters
    ----------
    time_series : array-like
        The input time series.
    lbound : int, optional
        The lower bound of the window size range. Default is 10.
    ubound : int, optional
        The upper bound of the window size range. Default is 1000.

    Returns
    -------
        The dominant Fourier frequency's corresponding window size within the specified range.
        period_condidates
        period_condidates_magnitudes
    """

    if time_series.shape[0] < 2 * lbound:
        warnings.warn(
            f"Time series must at least have 2*lbound much data points. Using window_size={time_series.shape[0]}.")
        return time_series.shape[0]

    fourier = np.fft.fft(time_series)
    freq = np.fft.fftfreq(time_series.shape[0], 1)

    magnitudes = []
    window_sizes = []

    for coef, freq in zip(fourier, freq):
        if coef and freq > 0:
            window_size = int(1 / freq)
            mag = math.sqrt(coef.real * coef.real + coef.imag * coef.imag)

            if window_size >= lbound and window_size < ubound:
                window_sizes.append(window_size)
                magnitudes.append(mag)

    if len(magnitudes) == 0:
        warnings.warn(f"Could not extract valid frequencies. Using window_size={lbound}.")
        return lbound
  
    return np.array(window_sizes)[np.argsort(magnitudes)[::-1]], np.sort(magnitudes)[::-1]


from difflib import SequenceMatcher
from collections import Counter


def calculate_similarity_score(strings_list):
    """
    Calculate an overall similarity score for a list of strings.
    The score is based on multiple similarity metrics.
    
    Args:
        strings_list: List of strings to compare
        
    Returns:
        float: Overall similarity score between 0 and 1
    """
    if not strings_list or len(strings_list) < 2:
        return 1.0  # A single string or empty list is perfectly similar to itself
    
    n = len(strings_list)
    total_comparisons = n * (n - 1) // 2
    
    # Initialize scores for different metrics
    sequence_scores = []
    jaccard_scores = []
    length_ratio_scores = []
    
    # Compare each pair of strings
    for i in range(n):
        for j in range(i + 1, n):
            str1 = strings_list[i]
            str2 = strings_list[j]
            
            # Sequence Matcher (difflib) score
            sequence_score = SequenceMatcher(None, str1, str2).ratio()
            sequence_scores.append(sequence_score)
            
            # Jaccard similarity (character-based)
            set1, set2 = set(str1), set(str2)
            jaccard_score = len(set1.intersection(set2)) / len(set1.union(set2)) if set1 or set2 else 1.0
            jaccard_scores.append(jaccard_score)
            
            # Length ratio (shorter/longer)
            length_ratio = min(len(str1), len(str2)) / max(len(str1), len(str2)) if max(len(str1), len(str2)) > 0 else 1.0
            length_ratio_scores.append(length_ratio)
    
    # Calculate average scores
    avg_sequence = np.mean(sequence_scores)
    avg_jaccard = np.mean(jaccard_scores)
    avg_length_ratio = np.mean(length_ratio_scores)
    
    # Calculate overall score (weighted average of the three metrics)
    overall_score = 0.5 * avg_sequence + 0.3 * avg_jaccard + 0.2 * avg_length_ratio
    
    return overall_score

def fuse_adjacent(s):
   if not s:
       return ''
   result = s[0]
   for c in s[1:]:
       if c != result[-1]:
           result += c
   return result


def find_longest_identical_pair(s):
    left = 0
    right = len(s) - 1
    id_pair = (None, -1, -1)
    while left < right:
        for i in range(left+1, right+1):
            if s[left] == s[i]:
                if id_pair[2] - id_pair[1] < i - left:
                    id_pair = (s[left], left, i)
        left += 1
    if id_pair[0] is None:
        return None  # If no identical pair is found
    else: 
        return id_pair


'''def number_to_alpha(numbers):
    # Create a mapping of numbers to alphabetic characters
    alpha_map = {i: chr(97 + i) for i in range(26)}  # a-z
    alpha_map.update({i + 26: chr(65 + i) for i in range(26)})  # A-Z
    
    # Convert numbers to characters
    result = ''
    for num in numbers:
        if num in alpha_map:
            result += alpha_map[num]
        else:
            result += '?'  # For numbers outside the range 0-51
    return result'''

def number_to_alpha(numbers):
    alpha_map = {i: chr(65 + i) for i in range(26)}  # A-Z
    
    result = ''
    for num in numbers:
        if num in alpha_map:
            result += alpha_map[num]
        else:
            result += '?'  # For numbers outside the range 0-25
    return result

def alpha_to_number(sequence):
   return [ord(c.upper()) - ord('A') for c in sequence]





def score_match(chars):
    """Score a column of aligned characters"""
    if '-' in chars:
        return -len([c for c in chars if c == '-'])  # Gap penalty
    return sum(1 for i, j in product(chars, chars) if i == j) - len(chars)  # Sum of pairwise matches

def initialize_matrix(sequences):
    """Initialize the N-dimensional DP matrix and pointers"""
    # Get dimensions for each sequence
    dims = [len(seq) + 1 for seq in sequences]
    
    # Create score matrix F and pointer matrix P
    F = np.zeros(dims)
    # Initialize P with lists instead of zeros
    P = np.empty(dims, dtype=object)
    for idx in np.ndindex(*dims):
        P[idx] = []
    
    # Initialize edges with gap penalties
    for idx, dim in enumerate(dims):
        # Create slice objects for each dimension
        slices = [slice(None) if i == idx else 0 for i in range(len(dims))]
        indices = range(1, dim)
        F[tuple(slices)] = np.linspace(0, -len(sequences) * dim, dim)
    
    return F, P

def get_neighbors(current_pos, dims):
    """Get all possible previous positions in the DP matrix"""
    neighbors = []
    for i in range(2 ** len(dims)):
        neighbor = []
        for j, pos in enumerate(current_pos):
            if i & (1 << j):
                if pos > 0:  # Check boundary
                    neighbor.append(pos - 1)
                else:
                    break
            else:
                neighbor.append(pos)
        if len(neighbor) == len(dims):
            neighbors.append(tuple(neighbor))
    return neighbors[1:]  # Exclude current position

def msa(sequences, gap_penalty=-1):
    """Perform multiple sequence alignment using N-dimensional Needleman-Wunsch"""
    # Initialize matrices
    F, P = initialize_matrix(sequences)
    dims = F.shape
    
    # Fill the DP matrix
    for pos in product(*[range(1, dim) for dim in dims]):
   
        # Get characters at current position
        chars = [sequences[i][pos[i]-1] for i in range(len(sequences))]
        
        # Get all possible previous positions
        neighbors = get_neighbors(pos, dims)
        
        # Calculate scores for all possible alignments
        max_score = float('-inf')
        best_moves = []
        
        for neighbor in neighbors:
            # Calculate score based on which sequences are aligned
            aligned_chars = []
            for i, (curr, prev) in enumerate(zip(pos, neighbor)):
                if curr != prev:
                    aligned_chars.append(sequences[i][curr-1])
                else:
                    aligned_chars.append('-')
            
            score = F[neighbor] + score_match(aligned_chars)
            
            if score > max_score:
                max_score = score
                best_moves = [neighbor]
            elif score == max_score:
                best_moves.append(neighbor)
        
        F[pos] = max_score
        P[pos] = best_moves  # Store list of best moves
    
    # Traceback
    aligned_sequences = [[] for _ in sequences]
    current_pos = tuple(dim-1 for dim in dims)
    
    while any(pos > 0 for pos in current_pos):
        # Ensure P[current_pos] contains valid moves
        if not P[current_pos]:  # If no moves stored, break
            break
            
        prev_pos = P[current_pos][0]  # Take first best move
        
        # Add characters or gaps based on moves
        for i, (curr, prev) in enumerate(zip(current_pos, prev_pos)):
            if curr != prev:
                aligned_sequences[i].append(sequences[i][curr-1])
            else:
                aligned_sequences[i].append('-')
        
        current_pos = prev_pos
    
    # Reverse and join sequences
    return [(''.join(seq))[::-1] for seq in aligned_sequences]


# Example usage
#sequences = ['ACBAFDECBAECFACBA', 'CFACBAFDECBAECFA', 'ECFACBAFBECBAECFA']
#aligned_sequences = msa(sequences)
#print('\n'.join(aligned_sequences))


def align_strings(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    max_matches = 0
    best_offset = 0
    for offset in range(-len2 + 1, len1):
        match_count = 0
        for i in range(len1):
            j = i - offset
            if 0 <= j < len2 and s1[i] == s2[j]:
                match_count += 1
        if match_count > max_matches:
            max_matches = match_count
            best_offset = offset
    return best_offset, max_matches


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
def align_multiple_strings(strings):
    if not strings:
        return []
    reference = strings[0]
    for next_string in strings[1:]:
        ref_align, _ = align_two_strings(reference, next_string)
        # Collapse underscores to match specification
        reference = ''.join(char for i, char in enumerate(ref_align) if char != '_' or (i > 0 and ref_align[i - 1] != '_'))
    return reference

def align_two_strings(str1, str2):
    # Alignment code (based on previously defined)
    n, m = len(str1), len(str2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1
    aligned1, aligned2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            aligned1.append(str1[i - 1])
            aligned2.append(str2[j - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned1.append('_')
            aligned2.append('_')
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j]+1:
            aligned1.append('_')
            i -= 1
        else:
            aligned2.append('_')
            j -= 1
    while i > 0:
        aligned1.append('_')
        i -= 1
    while j > 0:
        aligned2.append('_')
        j -= 1
    aligned1.reverse()
    aligned2.reverse()
    return ''.join(aligned1), ''.join(aligned2)
'''
