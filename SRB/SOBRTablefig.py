#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import os
from PIL import Image

# Define color values
color1 = '#FFFFFF'  # Unlikely color
color2 = '#F9C676'
color3 = '#FD8C1E'
color4 = '#EF0000'
color5 = '#70ACCE'
color6 = '#4065E8'
color7 = '#071BB7'
color8 = '#85a3b2'
color9 = 'teal'
color10='darkslategray'

# Define marker styles based on trend and significance
marker_styles_dict = {
    ('No trend', 'Unlikely'): 'o',
    ('Increasing', 'Unlikely'): 'o',
    ('Decreasing', 'Unlikely'): 'o',
    ('Increasing', 'Likely'): '^',
    ('Increasing', 'Very Likely'): '^',
    ('Increasing', 'Highly Likely'): '^',
    ('Decreasing', 'Likely'): 'v',
    ('Decreasing', 'Very Likely'): 'v',
    ('Decreasing', 'Highly Likely'): 'v'
}

# Define color mapping and edge color mapping for unlikely trends
color_mapping_unlikely = {
    ('No trend', 'Unlikely'): color1,
    ('Increasing', 'Unlikely'): color1,
    ('Decreasing', 'Unlikely'): color1,
}

edge_color_mapping_unlikely = {
    ('No trend', 'Unlikely'): 'black',
    ('Increasing', 'Unlikely'): 'black',
    ('Decreasing', 'Unlikely'): 'black',
}

# Define color mapping and edge color mapping for other trends
color_mapping = {
    ('Increasing', 'Likely'): color5,
    ('Increasing', 'Very Likely'): color6,
    ('Increasing', 'Highly Likely'): color7,
    ('Decreasing', 'Likely'): color2,
    ('Decreasing', 'Very Likely'): color3,
    ('Decreasing', 'Highly Likely'): color4
}

edge_color_mapping = {
    ('Increasing', 'Likely'): color5,
    ('Increasing', 'Very Likely'): color6,
    ('Increasing', 'Highly Likely'): color7,
    ('Decreasing', 'Likely'): color2,
    ('Decreasing', 'Very Likely'): color3,
    ('Decreasing', 'Highly Likely'): color4
}

# Define marker styles based on trend and significance for Trend SignificancePtt
markerpt_styles_dict = {
    'Unlikely': 'o',       # Circle
    'Likely': 'o',         # Star
    'Very Likely': 'o',    # Star
    'Highly Likely': 'o',  # Star
}

# Define color mapping and edge color mapping for Trend SignificancePtt
colorpt_mapping = {
    'Unlikely': color1,
    'Likely': color8,
    'Very Likely': color9,
    'Highly Likely': color10,
}

edge_colorpt_mapping = {
    'Unlikely': 'black',
    'Likely': color8,
    'Very Likely': color9,
    'Highly Likely': color10,
}

# def get_marker_style(trend_info, slope_percentage, min_slope, max_slope):
#     color_key = tuple(trend_info)
#     if color_key in color_mapping_unlikely:
#         color = color_mapping_unlikely[color_key]
#         edge_color = edge_color_mapping_unlikely[color_key]
#     else:
#         color = color_mapping.get(color_key, color1)
#         edge_color = edge_color_mapping.get(color_key, 'black')

#     marker = marker_styles_dict.get(color_key, 'o')

#     slope_distance = (max_slope - min_slope) / 3
#     if min_slope <= slope_percentage <= (min_slope + slope_distance):
#         marker_size = 15
#     elif (min_slope + slope_distance) < slope_percentage <= (min_slope + 2 * slope_distance):
#         marker_size = 20 * 2
#     else:
#         marker_size = 20 * 4

#     return marker, color, edge_color, marker_size

def get_marker_style(trend_info, slope_percentage, min_slope, max_slope):
    color_key = tuple(trend_info)
    if color_key in color_mapping_unlikely:
        color = color_mapping_unlikely[color_key]
        edge_color = edge_color_mapping_unlikely[color_key]
        marker = marker_styles_dict.get(color_key, 'o')
        marker_size = 15  # Ensure marker size remains 15 for "Unlikely" trends
    else:
        color = color_mapping.get(color_key, color1)
        edge_color = edge_color_mapping.get(color_key, 'black')

        marker = marker_styles_dict.get(color_key, 'o')

        slope_distance = (max_slope - min_slope) / 3
        if min_slope <= slope_percentage <= (min_slope + slope_distance):
            marker_size = 15
        elif (min_slope + slope_distance) < slope_percentage <= (min_slope + 2 * slope_distance):
            marker_size = 20 * 2
        else:
            marker_size = 20 * 4

    return marker, color, edge_color, marker_size



def scatter_trend_information(ax, subset_df, result_df, x_position):
    min_slope = min(result_df['Slope as % of MAF'])
    max_slope = max(result_df['Slope as % of MAF'])
    for station_name in subset_df['station']:
        matching_rows = result_df[result_df['Station'] == station_name]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            trend_info = (row['Trend Direction'], row['Trend Significance'])
            slope_percentage = row['Slope as % of MAF']
            marker, color, edge_color, marker_size = get_marker_style(trend_info, slope_percentage, min_slope, max_slope)
            ax.scatter(
                x_position,
                subset_df[subset_df['station'] == station_name]['Lat'].iloc[0],
                s=marker_size,
                marker=marker,
                color=color,
                edgecolor=edge_color,
                linewidth=0.25,
                label=station_name
            )
            trend_info_ptt = row['Trend SignificancePtt']
            markerpt = markerpt_styles_dict.get(trend_info_ptt, 'o')
            colorpt = colorpt_mapping.get(trend_info_ptt, color1)
            edge_colorpt = edge_colorpt_mapping.get(trend_info_ptt, 'black')
            marker_size_ptt = 15 if markerpt == 'o' else 80
            ax.scatter(
                x_position + 0.06,
                subset_df[subset_df['station'] == station_name]['Lat'].iloc[0],
                s=marker_size_ptt,
                marker=markerpt,
                color=colorpt,
                edgecolor=edge_colorpt,
                linewidth=0.25,
                label=station_name
            )

def annotate_stations(ax, subset_df):
    for i, txt in enumerate(subset_df['station']):
        ax.annotate(txt, (subset_df['Long'].iloc[i], subset_df['Lat'].iloc[i] - 0.25), fontsize=10)
        ax.axhline(y=subset_df['Lat'].iloc[i] - 0.5, color='grey', linestyle='-', linewidth=0.15)
        
        
def categorize_trend(ts, cs):
    if ts == 'Unlikely' and cs == 'Unlikely':
        return 'white', 'grey', 'o'  # Circle
    elif ts == 'Unlikely' and cs in ['Likely', 'Very Likely', 'Highly Likely']:
        return 'purple', 'purple', 's'  # Square
    elif ts in ['Likely', 'Very Likely', 'Highly Likely'] and cs == 'Unlikely':
        return 'green', 'green', 's'     # Diamond
    else:
        return 'red', 'red', 'D'  # Circle as default


def scatter_trend_significance(ax, subset_df, result_df, x_position):
    for _, row in subset_df.iterrows():
        station_name = row['station']
        matching_rows = result_df[result_df['Station'] == station_name]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            ts = row['Trend Significance']
            cs = row['Trend SignificancePtt']
            facecolor, edgecolor, marker = categorize_trend(ts, cs)
#             marker = 'D'  # Diamond marker
            ax.scatter(
                x_position,
                subset_df[subset_df['station'] == station_name]['Lat'].iloc[0],
                s=15,  # Marker size
                marker=marker,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.5,
                label=station_name
            )





def concat_images(folder_path):
    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sort the image files based on their names
    image_files.sort()

    # Initialize an empty list to store loaded and trimmed images
    loaded_images = []

    # Load each image with a name like "figure_1", "figure_2", etc.
    for i, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, image_file)
        
        try:
            # Open the image
            img = Image.open(image_path)
            
            # Trim white space around the image
            img = img.crop(img.getbbox())
            
            # Append the trimmed image to the list
            loaded_images.append(img)
            
            # Assign a name like "figure_1", "figure_2", etc.
            figure_name = f'figure_{i}'
            
            # You can use the 'figure_name' to refer to the loaded image in your code
            print(f"Loaded and trimmed image '{image_file}' as '{figure_name}'")
            
        except Exception as e:
            print(f"Error loading image '{image_file}': {e}")

    if not loaded_images:
        print("No images loaded. Exiting function.")
        return None

    # Determine the total height needed for concatenation
    total_height = sum(img.size[1] for img in loaded_images)

    # Create a new image with the width of the first image and the total height
    concatenated_image = Image.new('RGB', (loaded_images[0].size[0], total_height))

    # Paste each trimmed image vertically in the new image
    current_height = 0
    for img in loaded_images:
        concatenated_image.paste(img, (0, current_height))
        current_height += img.size[1]

    return concatenated_image

# Example usage:
# folder_path = '/path/to/your/folder'
# concatenated_image = concat_images(folder_path)
# concatenated_image.save('/path/to/save/concatenated_image.jpg')
