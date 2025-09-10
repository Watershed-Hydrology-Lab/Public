import matplotlib.pyplot as plt
import os
from PIL import Image
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import xclim
import os
import matplotlib.colors
import scipy.stats as st
import numpy as np
from scipy.stats import norm


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


def plot_trend_mapMK(results_df,shapefile_path,fig_folder1, title, filename):      
        # Extract the desired columns
        selected_columns = ['Station', 'Latitude', 'Longitude', 'Trend Direction', 'Trend Significance','Slope as % of MAF']
        selected_data = results_df[selected_columns]
#         selected_data.to_csv(os.path.join(fig_folder1, "selected_dataPTT.csv"), index=False)


        # Import results_df and base_map

        geometry = [Point(xy) for xy in zip(results_df['Longitude'], results_df['Latitude'])]
        gdf = gpd.GeoDataFrame(results_df, geometry=geometry)
        base_map = gpd.read_file(shapefile_path)
        
        
        

        # Convert the selected data to the desired format
        data = {
            'Station': selected_data['Station'].tolist(),
            'Latitude': selected_data['Latitude'].tolist(),
            'Longitude': selected_data['Longitude'].tolist(),
            'Trend Direction': selected_data['Trend Direction'].tolist(),
            'Trend Significance': selected_data['Trend Significance'].tolist(),
            'Slope as % of MAF': selected_data['Slope as % of MAF'].tolist()
        }

        # Import results_df and base_map

        geometry = [Point(xy) for xy in zip(results_df['Longitude'], results_df['Latitude'])]
        gdf = gpd.GeoDataFrame(results_df, geometry=geometry)


        # Define color values
        color1 = '#FFFFFF'  
        color2 = '#F9C676'
        color3 = '#FD8C1E'
        color4 = '#EF0000'
        color5 = '#70ACCE'
        color6 = '#4D7BFD'
        color7 = '#000F86'

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
        color_mapping= {
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

        # Define marker style function
        def marker_style(row):
            color_key = (row['Trend Direction'], row['Trend Significance'])

            if color_key in color_mapping_unlikely:
                color = color_mapping_unlikely[color_key]
                edge_color = edge_color_mapping_unlikely[color_key]
            else:
                color = color_mapping.get(color_key, color1)
                edge_color = edge_color_mapping.get(color_key, 'black')

            marker = marker_styles_dict.get(color_key, 'o')

            return {
                'color': color,
                'marker': marker,
                'edgecolor': edge_color,
                'linewidth': 0.25
            }

        # Apply the marker style function to the GeoDataFrame
        gdf['marker_styles'] = gdf.apply(marker_style, axis=1)

        # Create a plot
        fig, ax = plt.subplots(figsize=(8, 8))


        # Plot the base map
        # Create a colormap with two colors: white for Id=0 and blue for Id=1
        colors = ['white', '#DAE8F9']
        cmap = matplotlib.colors.ListedColormap(colors)

        # Plot the map using the custom colormap
        base_map.plot(column="Id", cmap=cmap, legend=False, edgecolor='black', linewidth=0.05, ax=ax)

        # Plot the region boundaries for the entire map
        base_map.boundary.plot(ax=ax, linewidth=0.005, color='gray')


        # Plot the points with different markers and colors based on the marker style for unlikely trends
        for idx, row in gdf[gdf['Trend Significance'] == 'Unlikely'].iterrows():
            ax.scatter(row['Longitude'], row['Latitude'], s=10, marker=row['marker_styles']['marker'],
                       color=row['marker_styles']['color'], edgecolor=row['marker_styles']['edgecolor'],
                       linewidth=row['marker_styles']['linewidth'])

            #Plot the points with different markers and colors based on the marker style for unlikely trends
        for idx, row in gdf[gdf['Trend Direction'] == 'No trend'].iterrows():
            ax.scatter(row['Longitude'], row['Latitude'], s=10, marker=row['marker_styles']['marker'],
                       color=row['marker_styles']['color'], edgecolor=row['marker_styles']['edgecolor'],
                       linewidth=row['marker_styles']['linewidth'])

          # Find the maximum and minimum slope values
        min_slope = min(data['Slope as % of MAF'])
        max_slope = max(data['Slope as % of MAF'])
        
       # Define three categories of equidistant % of slope_distance
        slope_distance = (max_slope - min_slope) / 3
        
        a= min_slope
        b= min_slope + slope_distance
        c= min_slope + 2*slope_distance


        # Update the marker size based on the 'Slope as % of MAF' column for non-unlikely trends
        for idx, row in gdf[(gdf['Trend Significance'] != 'Unlikely') & (gdf['Trend Direction'] != 'No trend')].iterrows():
            slope_percentage = row['Slope as % of MAF']

            if min_slope <= slope_percentage <= (min_slope + slope_distance):
                marker_size = 10  # Keep the size of the marker as it is
            elif (min_slope + slope_distance) < slope_percentage <= (min_slope + 2 * slope_distance):
                marker_size = 25   
            else:
                marker_size = 60 # Increase the size by 1.6

            ax.scatter(
                row['Longitude'],
                row['Latitude'],
                s=marker_size,
                marker=row['marker_styles']['marker'],
                color=row['marker_styles']['color'],
                edgecolor=row['marker_styles']['edgecolor'],
                linewidth=row['marker_styles']['linewidth']
            )

        ax.set_xticks([])
        ax.set_yticks([])

        # Add a legend based on trends and significance
        legend_labels = {
            'Increasing, Highly Likely': 'Increasing, Highly Likely',
            'Increasing, Very Likely': 'Increasing, Very Likely',
            'Increasing, Likely': 'Increasing, Likely',
            'No trend, Unlikely': 'Unlikely',
            'Decreasing, Likely': 'Decreasing, Likely',
            'Decreasing, Very Likely': 'Decreasing, Very Likely',
            'Decreasing, Highly Likely': 'Decreasing, Highly Likely',
        }


        legend_elements = [
            plt.Line2D([0], [0], marker=marker_styles_dict.get((label.split(', ')[0], label.split(', ')[1]), 'o'),
                       linestyle='None',
                       color=color_mapping.get((label.split(', ')[0], label.split(', ')[1]), color1),
                       markerfacecolor=color_mapping.get((label.split(', ')[0], label.split(', ')[1]), color1),
                       markeredgecolor='black' if marker_styles_dict.get((label.split(', ')[0], label.split(', ')[1])) == 'o' else color_mapping.get((label.split(', ')[0], label.split(', ')[1]), color1),  # Corrected the condition for circular marker edge color
                       markersize=6, markeredgewidth=0.5, label=legend_labels[label]) for label in legend_labels
        ]

        ax = plt.gca()

        # legend = ax.legend(handles=legend_elements, title=f'Trend Significance- \n Mean Annual Discharge \n ({period[0]}-{period[1]})', loc='upper right', frameon=False)
#         legend = ax.legend(handles=legend_elements, loc='upper right',handlelength=2, handletextpad=0.05, fontsize='9', frameon=False, title='3 Day Maximum Flow Timing\n ({0}-{1})\n Trend Significance'.format(str(periodd[0]), str(periodd[1])))
        legend = ax.legend(handles=legend_elements, loc='upper right', handlelength=2, handletextpad=0.05, fontsize='9', frameon=False, title=title)                           
        legend.get_title().set_fontsize(9) 
        legend.get_title().set_ha('center') 


        # Function to get the markersize from a Line2D object
        def get_markersize(line):
            return line._markersize

        # Custom marker
        spacing = 0.15
        # Create a custom marker with space between triangles
        custom_marker = r'$\bigtriangledown\!\!$' + ' ' * int(spacing * 10) + r'$\!\!\triangle$'

        triangle_legend_elements = [
            plt.Line2D([0], [0], marker=custom_marker, color='black', markerfacecolor='black', markersize=8, markeredgewidth=0.08, linestyle='None', label=f'{a:.2f} < %MAF ≤ {b:.2f}'),
            plt.Line2D([0], [0], marker=custom_marker, color='black', markerfacecolor='black', markersize=11, markeredgewidth=0.15, linestyle='None', label=f'{b:.2f} < %MAF ≤ {c:.2f}'),
            plt.Line2D([0], [0], marker=custom_marker, color='black', markerfacecolor='black', markersize=15, markeredgewidth=0.15, linestyle='None',  label=f'{c:.2f} < %MAF ≤ {max_slope:.2f}'),
        ]



        # Sort the legend elements based on size
        triangle_legend_elements.sort(key=get_markersize)

        # Plot the legend in the lower left with a centered title
        legend1 = plt.legend(handles=triangle_legend_elements, loc='lower left', handlelength=2, handletextpad=0.05, frameon=False)
        legend1.get_title().set_fontsize(9) 
        legend1.get_title().set_ha('center')
        # legend1.set_title("Size ∝ % Normalized \n Mean Annual flow", prop={'size': 'medium'})
        # legend1.set_title("Magnitude of Change", prop={'size': 'medium'})
        legend1.set_title("Magnitude of Change")
        plt.gca().add_artist(legend)

        # Set the font size for the legend labels
        for label in legend.get_texts():
            label.set_fontsize(7)

        # Set the font size for the legend labels
        for label in legend1.get_texts():
            label.set_fontsize(7)


        # Set the font size for the legend labels
        for label in legend.get_texts():
            label.set_fontsize(7)


       # Save the plot
        trend_path = os.path.join(fig_folder1, filename)
        plt.savefig(trend_path, dpi=1000,bbox_inches='tight')
        plt.show()
        


























def plot_trend_mapMKT(results_df,shapefile_path,fig_folder1, title, filename):      
        # Extract the desired columns
        selected_columns = ['Station', 'Latitude', 'Longitude', 'Trend Direction', 'Trend Significance','slope over CNP']
        selected_data = results_df[selected_columns]
#         selected_data.to_csv(os.path.join(fig_folder1, "selected_dataPTT.csv"), index=False)


        # Import results_df and base_map

        geometry = [Point(xy) for xy in zip(results_df['Longitude'], results_df['Latitude'])]
        gdf = gpd.GeoDataFrame(results_df, geometry=geometry)
        base_map = gpd.read_file(shapefile_path)
        
        
        

        # Convert the selected data to the desired format
        data = {
            'Station': selected_data['Station'].tolist(),
            'Latitude': selected_data['Latitude'].tolist(),
            'Longitude': selected_data['Longitude'].tolist(),
            'Trend Direction': selected_data['Trend Direction'].tolist(),
            'Trend Significance': selected_data['Trend Significance'].tolist(),
            'slope over CNP': selected_data['slope over CNP'].tolist()
        }

        # Import results_df and base_map

        geometry = [Point(xy) for xy in zip(results_df['Longitude'], results_df['Latitude'])]
        gdf = gpd.GeoDataFrame(results_df, geometry=geometry)


        # Define color values
        color1 = '#FFFFFF'  
        color2 = '#F9C676'
        color3 = '#FD8C1E'
        color4 = '#EF0000'
        color5 = '#70ACCE'
        color6 = '#4D7BFD'
        color7 = '#000F86'

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
        color_mapping= {
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

        # Define marker style function
        def marker_style(row):
            color_key = (row['Trend Direction'], row['Trend Significance'])

            if color_key in color_mapping_unlikely:
                color = color_mapping_unlikely[color_key]
                edge_color = edge_color_mapping_unlikely[color_key]
            else:
                color = color_mapping.get(color_key, color1)
                edge_color = edge_color_mapping.get(color_key, 'black')

            marker = marker_styles_dict.get(color_key, 'o')

            return {
                'color': color,
                'marker': marker,
                'edgecolor': edge_color,
                'linewidth': 0.25
            }

        # Apply the marker style function to the GeoDataFrame
        gdf['marker_styles'] = gdf.apply(marker_style, axis=1)

        # Create a plot
        fig, ax = plt.subplots(figsize=(8, 8))


        # Plot the base map
        # Create a colormap with two colors: white for Id=0 and blue for Id=1
        colors = ['white', '#DAE8F9']
        cmap = matplotlib.colors.ListedColormap(colors)

        # Plot the map using the custom colormap
        base_map.plot(column="Id", cmap=cmap, legend=False, edgecolor='black', linewidth=0.05, ax=ax)

        # Plot the region boundaries for the entire map
        base_map.boundary.plot(ax=ax, linewidth=0.005, color='gray')


        # Plot the points with different markers and colors based on the marker style for unlikely trends
        for idx, row in gdf[gdf['Trend Significance'] == 'Unlikely'].iterrows():
            ax.scatter(row['Longitude'], row['Latitude'], s=10, marker=row['marker_styles']['marker'],
                       color=row['marker_styles']['color'], edgecolor=row['marker_styles']['edgecolor'],
                       linewidth=row['marker_styles']['linewidth'])

            #Plot the points with different markers and colors based on the marker style for unlikely trends
        for idx, row in gdf[gdf['Trend Direction'] == 'No trend'].iterrows():
            ax.scatter(row['Longitude'], row['Latitude'], s=10, marker=row['marker_styles']['marker'],
                       color=row['marker_styles']['color'], edgecolor=row['marker_styles']['edgecolor'],
                       linewidth=row['marker_styles']['linewidth'])

          # Find the maximum and minimum slope values
        min_slope = min(data['slope over CNP'])
        max_slope = max(data['slope over CNP'])
        
       # Define three categories of equidistant % of slope_distance
        slope_distance = (max_slope - min_slope) / 3
        
        a= min_slope
        b= min_slope + slope_distance
        c= min_slope + 2*slope_distance


        # Update the marker size based on the 'slope over CNP' column for non-unlikely trends
        for idx, row in gdf[(gdf['Trend Significance'] != 'Unlikely') & (gdf['Trend Direction'] != 'No trend')].iterrows():
            slope_percentage = row['slope over CNP']

            if min_slope <= slope_percentage <= (min_slope + slope_distance):
                marker_size = 10  # Keep the size of the marker as it is
            elif (min_slope + slope_distance) < slope_percentage <= (min_slope + 2 * slope_distance):
                marker_size = 25   
            else:
                marker_size = 60 # Increase the size by 1.6

            ax.scatter(
                row['Longitude'],
                row['Latitude'],
                s=marker_size,
                marker=row['marker_styles']['marker'],
                color=row['marker_styles']['color'],
                edgecolor=row['marker_styles']['edgecolor'],
                linewidth=row['marker_styles']['linewidth']
            )

        ax.set_xticks([])
        ax.set_yticks([])

        # Add a legend based on trends and significance
        legend_labels = {
            'Increasing, Highly Likely': 'Later, Highly Likely',
            'Increasing, Very Likely': 'Later, Very Likely',
            'Increasing, Likely': 'Later, Likely',
            'No trend, Unlikely': 'Unlikely',
            'Decreasing, Likely': 'Earlier, Likely',
            'Decreasing, Very Likely': 'Earlier, Very Likely',
            'Decreasing, Highly Likely': 'Earlier, Highly Likely',
        }


        legend_elements = [
            plt.Line2D([0], [0], marker=marker_styles_dict.get((label.split(', ')[0], label.split(', ')[1]), 'o'),
                       linestyle='None',
                       color=color_mapping.get((label.split(', ')[0], label.split(', ')[1]), color1),
                       markerfacecolor=color_mapping.get((label.split(', ')[0], label.split(', ')[1]), color1),
                       markeredgecolor='black' if marker_styles_dict.get((label.split(', ')[0], label.split(', ')[1])) == 'o' else color_mapping.get((label.split(', ')[0], label.split(', ')[1]), color1),  # Corrected the condition for circular marker edge color
                       markersize=6, markeredgewidth=0.5, label=legend_labels[label]) for label in legend_labels
        ]

        ax = plt.gca()

        # legend = ax.legend(handles=legend_elements, title=f'Trend Significance- \n Mean Annual Discharge \n ({period[0]}-{period[1]})', loc='upper right', frameon=False)
#         legend = ax.legend(handles=legend_elements, loc='upper right',handlelength=2, handletextpad=0.05, fontsize='9', frameon=False, title='3 Day Maximum Flow Timing\n ({0}-{1})\n Trend Significance'.format(str(periodd[0]), str(periodd[1])))
        legend = ax.legend(handles=legend_elements, loc='upper right', handlelength=2, handletextpad=0.05, fontsize='9', frameon=False, title=title)                           
        legend.get_title().set_fontsize(9) 
        legend.get_title().set_ha('center') 


        # Function to get the markersize from a Line2D object
        def get_markersize(line):
            return line._markersize

        # Custom marker
        spacing = 0.15
        # Create a custom marker with space between triangles
        custom_marker = r'$\bigtriangledown\!\!$' + ' ' * int(spacing * 10) + r'$\!\!\triangle$'

        triangle_legend_elements = [
            plt.Line2D([0], [0], marker=custom_marker, color='black', markerfacecolor='black', markersize=8, markeredgewidth=0.08, linestyle='None', label=f'{a:.2f} < CLINO ≤ {b:.2f}'),
            plt.Line2D([0], [0], marker=custom_marker, color='black', markerfacecolor='black', markersize=11, markeredgewidth=0.15, linestyle='None', label=f'{b:.2f} < CLINO≤ {c:.2f}'),
            plt.Line2D([0], [0], marker=custom_marker, color='black', markerfacecolor='black', markersize=15, markeredgewidth=0.15, linestyle='None',  label=f'{c:.2f} < CLINO ≤ {max_slope:.2f}'),
        ]



        # Sort the legend elements based on size
        triangle_legend_elements.sort(key=get_markersize)

        # Plot the legend in the lower left with a centered title
        legend1 = plt.legend(handles=triangle_legend_elements, loc='lower left', handlelength=2, handletextpad=0.05, frameon=False)
        legend1.get_title().set_fontsize(9) 
        legend1.get_title().set_ha('center')
        # legend1.set_title("Size ∝ % Normalized \n Mean Annual flow", prop={'size': 'medium'})
        # legend1.set_title("Magnitude of Change", prop={'size': 'medium'})
        legend1.set_title("Magnitude of Change")
        plt.gca().add_artist(legend)

        # Set the font size for the legend labels
        for label in legend.get_texts():
            label.set_fontsize(7)

        # Set the font size for the legend labels
        for label in legend1.get_texts():
            label.set_fontsize(7)


        # Set the font size for the legend labels
        for label in legend.get_texts():
            label.set_fontsize(7)


       # Save the plot
        trend_path = os.path.join(fig_folder1, filename)
        plt.savefig(trend_path, dpi=1000,bbox_inches='tight')
        plt.show()
        


def plot_trend_mapPtt(results_df,shapefile_path,fig_folder1, title, filename):
      # Extract the desired columns
        selected_columns = ['Station', 'Latitude', 'Longitude', 'Trend SignificancePtt']
        selected_data = results_df[selected_columns]
#         selected_data.to_csv(os.path.join(fig_folder1, "selected_dataCHANG.csv"), index=False)

        #  imported results_df and base_map

        geometry = [Point(xy) for xy in zip(results_df['Longitude'], results_df['Latitude'])]
        gdf = gpd.GeoDataFrame(results_df, geometry=geometry)
        base_map = gpd.read_file(shapefile_path)

        # Convert the selected data to the desired format
        data = {
            'Station': selected_data['Station'].tolist(),
            'Latitude': selected_data['Latitude'].tolist(),
            'Longitude': selected_data['Longitude'].tolist(),
            'Trend SignificancePtt': selected_data['Trend SignificancePtt'].tolist()
        }


        # Define marker styles based on trend and significance
        marker_styles_dict = {
            ( 'Unlikely'): 'o',       # Circle
            ('Likely'): 'o',       # Upward arrow
            ('Very Likely'): 'o',  # Upward arrow
            ('Highly Likely'): 'o', # Upward arrow 
        }

        # Define color mapping and edge color mapping
        color_mapping = {
            ('Unlikely'): color1,
            ('Likely'): color8,
            ('Very Likely'): color9,
            ('Highly Likely'): color10,
        }

        edge_color_mapping = {
            ('Unlikely'): 'black',
            ('Likely'): color8,
            ('Very Likely'): color9,
            ('Highly Likely'): color10,
        }

        # Define marker style function
        def marker_style(row):
            color_key = row['Trend SignificancePtt']  # Adjusted the order of keys

            # Default to color1 if the key is not found
            color = color_mapping.get(color_key, color1)
            edge_color_key = row['Trend SignificancePtt']
            edge_color = edge_color_mapping.get(edge_color_key, color1)  # Use edge color mapping

            marker = marker_styles_dict.get(color_key, 'o')  # Adjusted the key for marker_styles_dict

            return {
                'color': color,
                'marker': marker,
                'edgecolor': edge_color,  # Set edge color based on mapping
                'linewidth': 0.25
            }

        # Apply the marker style function to the GeoDataFrame
        gdf['marker_styles'] = gdf.apply(marker_style, axis=1)

        # Create a plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot the base map
        # Create a colormap with two colors: white for Id=0 and blue for Id=1
        colors = ['white', '#DAE8F9']
        cmap = matplotlib.colors.ListedColormap(colors)

        # Plot the map using the custom colormap
        base_map.plot(column="Id", cmap=cmap, legend=False, edgecolor='black', linewidth=0.05, ax=ax)

        # Plot the region boundaries for the entire map
        base_map.boundary.plot(ax=ax, linewidth=0.005, color='gray')

        # # Plot the points with different markers and colors
        # for idx, row in gdf.iterrows():
        #     ax.scatter(row['Longitude'], row['Latitude'], s=20, marker=row['marker_styles']['marker'], color=row['marker_styles']['color'], edgecolor=row['marker_styles']['edgecolor'], linewidth=row['marker_styles']['linewidth'])

        # Plot the points with different markers and colors based on the marker style for unlikely trends
        for idx, row in gdf[gdf['Trend SignificancePtt'] == 'Unlikely'].iterrows():
            ax.scatter(row['Longitude'], row['Latitude'], s=8, marker=row['marker_styles']['marker'],
                       color=row['marker_styles']['color'], edgecolor=row['marker_styles']['edgecolor'],
                       linewidth=row['marker_styles']['linewidth'])

        #     #Plot the points with different markers and colors based on the marker style for unlikely trends
        # for idx, row in gdf[gdf['Trend Direction'] == 'No trend'].iterrows():
        #     ax.scatter(row['Longitude'], row['Latitude'], s=10, marker=row['marker_styles']['marker'],
        #                color=row['marker_styles']['color'], edgecolor=row['marker_styles']['edgecolor'],
        #                linewidth=row['marker_styles']['linewidth'])

        # Plot the points with different markers and colors based on the marker style for other trends
        for idx, row in gdf[gdf['Trend SignificancePtt'] != 'Unlikely'].iterrows():
            ax.scatter(row['Longitude'], row['Latitude'], s=18, marker=row['marker_styles']['marker'],
                       color=row['marker_styles']['color'], edgecolor=row['marker_styles']['edgecolor'],
                       linewidth=row['marker_styles']['linewidth'])  



        legend_labels = {
            'Highly Likely': 'Highly Likely-change ',
            'Very Likely': 'Very Likely-change ',
            'Likely': 'Likely-change ',
            'Unlikely': 'No Change',
        }
        ax.set_xticks([])
        ax.set_yticks([])

        legend_elements = [
            plt.Line2D([0], [0], marker=marker_styles_dict[label],
                       linestyle='None',
                       color=color_mapping[label],
                       markerfacecolor=color_mapping[label],
                       markeredgecolor=edge_color_mapping[label],
                       markersize=6, markeredgewidth=0.5,label=legend_labels[label]) for label in legend_labels
        ]

        # Rest of your code remains unchanged

        ax = plt.gca()
        # legend = ax.legend(handles=legend_elements, title='Change Significance ', loc='upper right', frameon=False)
        legend = ax.legend(handles=legend_elements, loc='upper right', handlelength=2, handletextpad=0.05, fontsize='9', frameon=False, title=title)                                 

        legend.get_title().set_fontsize(9)  # Adjust title font size
        legend.get_title().set_ha('center') 

        # Set the font size for the legend labels
        for label in legend.get_texts():
            label.set_fontsize(7)


        trend_path = os.path.join(fig_folder1,  filename)
        plt.savefig(trend_path, dpi=1000,bbox_inches='tight')
        plt.show()
        
