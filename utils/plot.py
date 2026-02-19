import matplotlib.pyplot as plt
import numpy as np
from colorsys import hsv_to_rgb
import string
import networkx as nx
import re

import osmnx as ox
import matplotlib.animation as animation
from itertools import combinations
import random


#################plot for transcripts############################
def plot_string(text, figsize=(18, 4)):
    """
    Plot string with identical colors for identical letters and 0.2x letter width spacing.
    Supports lowercase letters, uppercase letters, and underscores.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=figsize)
    
    # Include underscore in the character set
    unique_chars = (sorted(set(string.ascii_lowercase))[:10])
    hues = np.linspace(0, 1, len(unique_chars), endpoint=False)
    color_map = {char: hsv_to_rgb(hue, 0.8, 0.9) 
                 for char, hue in zip(unique_chars, hues)}
    unique_chars += ['-']
    # Add special handling for underscore
    color_map['-'] = (0.5, 0.5, 0.5)  # Gray color for underscore

    unique_chars += (sorted(set(string.ascii_lowercase))[10:])
    color_map = {char: hsv_to_rgb(hue, 0.3, 0.8) 
                 for char, hue in zip(unique_chars, hues)}
    
    spacing = 0.1  # Space between letters relative to letter width
    width = 1.0    # Width of each letter
    total_width = len(text) * width * (1 + spacing) - spacing
    
    for i, char in enumerate(text):
        x_pos = i * width * (1 + spacing)
        
        if char == '_':
            # Draw underscore as a line slightly below the baseline
            plt.plot([x_pos - width/3, x_pos + width/3], 
                    [-0.2, -0.2], 
                    color=color_map['_'], 
                    linewidth=2)
        else:
            # Regular character plotting
            color = color_map[char.lower()]
            plt.text(x_pos, 0, char, fontsize=14, color=color,
                    ha='center', va='center')
    
    plt.xlim(-width/2, total_width - width/2)
    plt.ylim(-0.5, 0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('transcript.svg', format='svg')
    #plt.show()

# Example usage
#text =  "AACCCBBAAFFDDDEEEECCCCBBAAAEEECCCFFFFAAACCBBAAFFDDDDEEEEECCCCCBBBBAAAEECCCCCFFFAAAACCCBBBBAAAFFFBBBEECCBBBAAEECCCFFAA"
#plot_string(text)




def parse_sequence(sequence):
    """Parse sequence into main path and branches"""
    branches = []
    main_path = []
    current_branch = []
    in_branch = False
    
    for part in sequence.split(','):
        part = part.strip()
        if '(' in part:
            in_branch = True
            part = part.replace('(', '')
        if ')' in part:
            in_branch = False
            part = part.replace(')', '')
        
        nodes = re.findall(r'([A-Za-z_]\d+)', part)
        if len(nodes) == 2:
            if in_branch:
                current_branch.extend(nodes)
            else:
                if current_branch:
                    branches.append(current_branch)
                    current_branch = []
                main_path.extend(nodes)
    
    if current_branch:
        branches.append(current_branch)
    
    return main_path, branches

def get_node_number(node):
    """Extract number from node label"""
    return int(re.findall(r'\d+', node)[0])

def plot_sequence(sequence, figsize=(8, 3)):
    plt.figure(figsize=figsize)

    # Assign colors
    unique_chars = sorted(set(string.ascii_uppercase))[:10]
    hues = np.linspace(0, 1, len(unique_chars), endpoint=False)
    color_map = {char: hsv_to_rgb(hue, 0.8, 0.9) 
                 for char, hue in zip(unique_chars, hues)}

    unique_chars += ['_']
    # Add special handling for underscore
    color_map['_'] = (0.5, 0.5, 0.5)  # Gray color for underscore

    color_map.update({char: hsv_to_rgb(hue, 0.3, 0.8) 
                 for char, hue in zip(sorted(set(string.ascii_uppercase))[10:], hues)})
    
    main_path, branches = parse_sequence(sequence)
    G = nx.DiGraph()
    
    # Calculate positions
    pos = {}
    x_spacing = 1
    y_spacing = 0.5
    
    # Group nodes by their number
    nodes_by_number = {}
    all_nodes = set(main_path)
    for branch in branches:
        all_nodes.update(branch)
    
    for node in all_nodes:
        num = get_node_number(node)
        if num not in nodes_by_number:
            nodes_by_number[num] = []
        nodes_by_number[num].append(node)
    
    # Position nodes
    for num in sorted(nodes_by_number.keys()):
        nodes = nodes_by_number[num]
        x = (num - 1) * x_spacing
        
        if len(nodes) == 1:
            pos[nodes[0]] = (x, 0)
        else:
            # Center branching nodes vertically
            total_height = (len(nodes) - 1) * y_spacing
            start_y = -total_height / 2
            for i, node in enumerate(sorted(nodes)):
                pos[node] = (x, start_y + i * y_spacing)
    
    # Add edges
    for i in range(0, len(main_path)-1, 2):
        G.add_edge(main_path[i], main_path[i+1])
    
    for branch in branches:
        for i in range(0, len(branch)-1, 2):
            G.add_edge(branch[i], branch[i+1])
    

    # Draw arrows
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrowsize=10, width=1.5,
                          arrowstyle='->')
    
    # Draw nodes
    for node in G.nodes():
        letter = node[0]
        color = color_map[letter]

        circle = plt.Circle(pos[node], 0.1,
                          color=color, alpha=0.3)
        plt.gca().add_patch(circle)
        
        plt.text(pos[node][0], pos[node][1], node[0],
                color=color, fontsize=8,
                ha='center', va='center',
                fontweight='bold')
    
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('transcript.svg', format='svg')

# Example usage
#sequence =  "A1->C2, C2->B3, B3->A4, A4->F5, (F5->B6, F5->D6), (B6->E7, D6->E7), E7->C8, C8->B9, B9->A10, A10->E11, E11->C12, C12->F13"
#plot_sequence(sequence, (5.4,2))

#################plot for transcripts############################



#################plot for 2D trajectories############################





def plot_routes_animation(G, routes, colors, output_file, fps=20, duration_sec=10):
    """
    Create an animation showing routes appearing dynamically.
    
    Args:
        G (networkx.MultiDiGraph): Street network graph
        routes (list): List of routes (each route is a list of nodes)
        colors (list): List of colors for each route
        output_file (str): Output filename (should end with .gif or .mp4)
        fps (int): Frames per second
        duration_sec (int): Total animation duration in seconds
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Plot the base map
    ox.plot_graph(G, ax=ax, show=False, close=False,
                 edge_color='gray', edge_alpha=0.2, node_size=0)
    
    # Create empty route lines
    route_lines = []
    route_points = []
    
    # Initialize all routes as empty
    for color in colors:
        line, = ax.plot([], [], marker='D', color=color, linewidth=2, alpha=0.8, zorder=2)
        route_lines.append(line)
        route_points.append([])
    
    # Extract coordinates for all routes
    all_route_coords = []
    for route in routes:
        coords = []
        for node in route:
            x = G.nodes[node]['x']
            y = G.nodes[node]['y']
            coords.append((x, y))
        all_route_coords.append(coords)
    
    # Calculate total number of frames
    total_frames = fps * duration_sec
    
    # Animation update function
    def update(frame):
        # Calculate progress (0 to 1)
        progress = frame / total_frames
        
        # Update each route
        for i, coords in enumerate(all_route_coords):
            # Determine how many points to show for this route
            route_progress = min(1.0, progress * len(routes) - i)
            
            if route_progress <= 0:
                # Route hasn't started yet
                route_lines[i].set_data([], [])
                continue
                
            # Calculate number of points to show
            num_points = max(2, int(route_progress * len(coords)))
            
            # Get coordinates to display
            visible_coords = coords[:num_points]
            xs, ys = zip(*visible_coords) if visible_coords else ([], [])
            
            # Update line data
            route_lines[i].set_data(xs, ys)
        
        # Update legend based on which routes are visible
        visible_routes = [i for i, line in enumerate(route_lines) 
                         if len(line.get_xdata()) > 0]
        
        if visible_routes:
            # Update legend with only visible routes
            ax.legend([route_lines[i] for i in visible_routes],
                     [f'Period {i+1}' for i in visible_routes],
                     loc='upper right', prop={'size': 14},
                     bbox_to_anchor=(1, 1))
        
        return route_lines
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000/fps, blit=True
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save animation
    if output_file.endswith('.gif'):
        ani.save(output_file, writer='pillow', fps=fps, dpi=150)
    else:
        # For MP4, use ffmpeg
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        ani.save(output_file, writer=writer, dpi=150)
    
    plt.close()
    
    print(f"Animation saved to {output_file}")


# Example usage:
# plot_routes_animation(G, routes, colors, "route_animation.gif")
# For MP4: plot_routes_animation(G, routes, colors, "route_animation.mp4")


def plot_routes(G, routes, colors, output_file):
    """
    Plot multiple routes on the same map.
    
    Args:
        G (networkx.MultiDiGraph): Street network graph
        routes (list): List of routes (each route is a list of nodes)
        colors (list): List of colors for each route
        output_file (str): Output filename
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Plot the base map
    ox.plot_graph(G, ax=ax, show=False, close=False,
                 edge_color='gray', edge_alpha=0.2, node_size=0)
    
    # Create empty list to store route lines for legend
    route_lines = []
    
    # Plot each route
    for route, color in zip(routes, colors):
        # Extract the coordinates for each node in the route
        xs = []
        ys = []
        for node in route:
            # Get node coordinates
            x = G.nodes[node]['x']
            y = G.nodes[node]['y']
            xs.append(x)
            ys.append(y)
        
        # Plot the route
        line = ax.plot(xs, ys, marker='D', color=color, linewidth=2, alpha=0.2, zorder=2)[0]
        route_lines.append(line)
    
    # Add legend
    ax.legend(route_lines, 
             [f'Period {i+1}' for i in range(len(routes))],
             loc='upper right', prop={'size': 14},
             bbox_to_anchor=(1.25, 0.85))
    
    # Adjust layout and save
    plt.tight_layout()
    
    
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()




#################plot for 2D trajectories############################


def plot_task_2(obs_len, gt_seq_len, pred_seq_len, figsize_w=10, title=None):
    """
    Plot both GT and Pred timelines in the same figure with aligned scales.
    
    Args:
        obs_len: Length of observation period
        gt_seq_len: Total sequence length for ground truth
        pred_seq_len: Total sequence length for prediction
        figsize_w: Width of the figure
        title: Optional title for the figure
    """
    # Use the maximum sequence length to determine the x-axis limits
    max_seq_len = max(gt_seq_len, pred_seq_len)
    
    # Create figure with two subplots, one for GT and one for Pred
    fig, axes = plt.subplots(2, 1, figsize=(figsize_w, 2.5), gridspec_kw={'hspace': 0.3})
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Create consistent bar heights and label offset
    bar_height = 0.5
    label_offset = max_seq_len * 0.15  # Proportional offset based on sequence length
    
    # GT plot (top)
    y_position = 0
    axes[0].barh(y_position, obs_len, height=bar_height, left=0, color='lightgray')
    axes[0].barh(y_position, gt_seq_len+1-obs_len, height=bar_height, left=obs_len, color='lightgreen')
    axes[0].text(-label_offset, y_position, "GT:", fontsize=12, fontweight='bold', verticalalignment='center')
    
    # Pred plot (bottom)
    axes[1].barh(y_position, obs_len, height=bar_height, left=0, color='lightgray')
    axes[1].barh(y_position, pred_seq_len+1-obs_len, height=bar_height, left=obs_len, color='lightblue')
    axes[1].text(-label_offset, y_position, "Pred:", fontsize=12, fontweight='bold', verticalalignment='center')
    
    # Configure both axes consistently
    for ax in axes:
        # Set consistent x-limits for alignment
        ax.set_xlim(-label_offset, max_seq_len+1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        
        # Remove the box/frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add a thin line below the bar for better visibility
        ax.axhline(y_position - bar_height/2, color='black', linewidth=0.5)
    
    # Set tick marks for each plot
    axes[0].set_xticks([0, obs_len, gt_seq_len])
    axes[1].set_xticks([0, obs_len, pred_seq_len])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])
    return fig


def plot_task_3(gt_seq_len, GT_start, GT_end, pred_start, pred_end, figsize_w=10, title=None):
    """
    Plot both GT and Pred timelines in the same figure with aligned scales.
    
    Args:
        gt_seq_len: Total sequence length for ground truth
        GT_start: Start position of GT highlight bar
        GT_end: End position of GT highlight bar
        pred_start: Start position of prediction highlight bar
        pred_end: End position of prediction highlight bar
        figsize_w: Width of the figure
        title: Optional title for the figure
    """
    # Use the maximum sequence length to determine the x-axis limits
    max_seq_len = gt_seq_len
    
    # Create figure with two subplots, one for GT and one for Pred
    fig, axes = plt.subplots(2, 1, figsize=(figsize_w, 2.5), gridspec_kw={'hspace': 0.3})
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Create consistent bar heights and label offset
    bar_height = 0.5
    label_offset = max_seq_len * 0.15  # Proportional offset based on sequence length
    
    # GT plot (top)
    y_position = 0
    # Plot full lightgray bar for GT
    axes[0].barh(y_position, gt_seq_len, height=bar_height, left=0, color='lightgray')
    # Plot lightgreen bar within GT from GT_start to GT_end
    axes[0].barh(y_position, GT_end - GT_start, height=bar_height, left=GT_start, color='lightgreen')
    axes[0].text(-label_offset, y_position, "GT:", fontsize=12, fontweight='bold', verticalalignment='center')
    
    # Pred plot (bottom)
    # Plot full lightgray bar for Pred
    axes[1].barh(y_position, gt_seq_len, height=bar_height, left=0, color='lightgray')
    # Plot lightblue bar within Pred from pred_start to pred_end
    axes[1].barh(y_position, pred_end - pred_start, height=bar_height, left=pred_start, color='lightblue')
    axes[1].text(-label_offset, y_position, "Pred:", fontsize=12, fontweight='bold', verticalalignment='center')
    
    # Configure both axes consistently
    for ax in axes:
        # Set consistent x-limits for alignment
        ax.set_xlim(-label_offset, max_seq_len+1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        
        # Remove the box/frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add a thin line below the bar for better visibility
        ax.axhline(y_position - bar_height/2, color='black', linewidth=0.5)
    
    # Set tick marks for each plot
    axes[0].set_xticks([0, GT_start, GT_end, gt_seq_len])
    axes[1].set_xticks([0, pred_start, pred_end, gt_seq_len])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])
    return fig


import string
def plot_images_with_token(images, tokens, n_rows = 2):
    assert len(images) == len(tokens), "Each image must have a corresponding token"

    n_images = len(images)
    # Calculate rows and columns for grid layout
    
    n_cols = (n_images + 1) // n_rows  # Ceiling division to handle odd number of images

    # Create a figure to display the images
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    plt.rcParams['font.family'] = 'Times New Roman'
    unique_chars = (sorted(set(string.ascii_lowercase))[:10])
    hues = np.linspace(0, 1, len(unique_chars), endpoint=False)
    color_map = {char: hsv_to_rgb(hue, 0.8, 0.9) 
                 for char, hue in zip(unique_chars, hues)}

    # Make axes a 2D array even if there's just one column
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easy iteration if there are multiple columns
    axes_flat = axes.flatten()
    
    for i, (image, token) in enumerate(zip(images, tokens)):
        if i < len(axes_flat):
            color = color_map[token.lower()]
            axes_flat[i].imshow(image)
            axes_flat[i].set_title(token, color=color, size=50)
            axes_flat[i].axis('off')  # Hide axes
    
    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        if j < len(axes_flat):
            axes_flat[j].axis('off')
            fig.delaxes(axes_flat[j])
            
    plt.tight_layout()
    plt.savefig('anchors.jpg', bbox_inches='tight', pad_inches=0)
    #plt.show()


