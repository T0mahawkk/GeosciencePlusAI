import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
import datetime


class VisualiseWellData():

    def __init__(self):
        
        self.lithology_labels = {
            'Sandstone': {'color': '#ffff00', 'hatch': '..'},
            'Marl': {'color': '#80ffff', 'hatch': ''}, 
            'Limestone': {'color': '#4682B4', 'hatch': '++'},
            'Coal': {'color': 'black', 'hatch': ''},
            'Silt': {'color': '#7cfc00', 'hatch': '||'},
            'Claystone': {'color': '#228B22', 'hatch': '--'}  
        }

    def visualise_lithology_distribution(self, csv_file_path, well_name):
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Check if 'LITHOLOGY' column exists
        if 'LITHOLOGY' not in df.columns:
            print("Column 'LITHOLOGY' not found in the CSV file.")
            return
        
        # Get the lithology distribution
        lithology_counts = df['LITHOLOGY'].value_counts()
        
        # Dictionary of lithology properties (color, hatch symbol)
        lithology_dict = self.lithology_labels
        
        # Plot the distribution with the assigned colors and hatches
        fig, ax = plt.subplots(figsize=(8, 6))

        bars = []
        for lithology, count in lithology_counts.items():
            color = lithology_dict.get(lithology, {}).get('color', '#D2B48C')  # Default color if lithology not in dict
            hatch = lithology_dict.get(lithology, {}).get('hatch', '')  # Default hatch if not defined
            
            bar = ax.bar(lithology, count, color=color, hatch=hatch)
            bars.append(bar)

        # Add counts above the bars
        for bar in bars:
            for rect in bar:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), str(int(rect.get_height())), ha='center', va='bottom', fontsize=10)

        # Create custom legend handles
        legend_handles = [
            Patch(facecolor=lithology_dict[lithology]["color"], hatch=lithology_dict[lithology]["hatch"], label=lithology)
            for lithology in lithology_dict
        ]
        
        # Add the legend
        ax.legend(handles=legend_handles, title='Lithology', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add labels and title
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlabel('Lithology', fontsize=10)
        ax.set_title(f'Lithology Distribution for Well {well_name}', fontsize=12)
        plt.xticks(rotation=45, ha='right')  # Adjust x-axis labels for better readability
        
        plt.tight_layout()  # Adjust layout for better fit
        plt.show()
        
    def show_available_logs(self, csv_file_path):
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Check if the file has any data
        if df.empty:
            print("The CSV file is empty.")
            return
        
        # Show the column names (which represent the available logs)
        print("Available logs in the data file:")
        
        # Iterate through each column and print statistics
        for column in df.columns:

            if column == 'LITHOLOGY':  # Skip the 'LITHOLOGY' column
                continue
                
            print(f"\nStatistics for '{column}':")
            column_data = df[column]
            
            # Calculate statistics
            count = column_data.count()  # Number of non-NA/null entries
            mean = column_data.mean()  # Mean of the column
            std_dev = column_data.std()  # Standard deviation
            min_val = column_data.min()  # Minimum value
            max_val = column_data.max()  # Maximum value

            # Print the statistics
            print(f"  Count: {count}")
            print(f"  Mean: {mean:.3f}")
            print(f"  Standard Deviation: {std_dev:.3f}")
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")

    def crossplot_2D(self, csv_file_path, well_name, x_col, y_col, x_in_log=False, y_in_log=False):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Check if the selected columns exist
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Columns '{x_col}' or '{y_col}' not found in the CSV file.")
            return
    
        # Handle negative values for the selected columns (if they exist)
        for col in [x_col, y_col]:
            if (df[col] < 0).any():
                negative_count = (df[col] < 0).sum()
                df[col] = df[col].clip(lower=0)  # Clip negative values to 0
                print(f"{negative_count} negative values in '{col}' have been clipped to 0.")
            
        # Drop rows with NaN values in the selected columns
        df = df.dropna(subset=[x_col, y_col])
    
        # Extract the data for plotting
        x_data = df[x_col].values.reshape(-1, 1)
        y_data = df[y_col].values
    
        # Linear regression
        model = LinearRegression()
        model.fit(x_data, y_data)
        y_pred = model.predict(x_data)
    
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_data, y_data, label='Data points', color='blue', alpha=0.5)
        ax.plot(x_data, y_pred, color='red', linewidth=2, label='Linear fit')
    
        # Apply log scale based on function input
        if x_in_log:
            ax.set_xscale("log")
        if y_in_log:
            ax.set_yscale("log")
    
        # Add labels and title
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.set_title(f'{x_col} vs {y_col} for Well {well_name}', fontsize=12)
        ax.legend()

        plt.tight_layout()  # Adjust layout for better fit
        plt.show()

        # Example usage:
        # visualiser.crossplot_2D(r".\Well Data\15_9-F-14.csv", "15_9-F-14", "NPHI", "RHOB")

    def plot_well_logs_and_lithology(self, csv_file_path, well_name, track_config=None, log_limits=None, 
                                    figure_width=None, figure_height=None, color_scheme=None, vertical_exaggeration=500):
        """
        Plots well logs and lithology from a CSV file with dynamically optimized layout and improved visualization.
        
        Parameters:
        - csv_file_path: str, path to the CSV file containing well log data.
        - well_name: str, name of the well (used for plot title).
        - track_config: dict, optional configuration for tracks. Keys are track names, values are lists of log names.
        - log_limits: dict, optional limits for each log. Keys are log names, values are tuples of (min, max).
        - figure_width: float, optional width of the figure in inches. If None, calculated automatically.
        - figure_height: float, optional height of the figure in inches. If None, calculated based on vertical_exaggeration.
        - color_scheme: str, optional color scheme to use. Options: 'default', 'viridis', 'plasma', 'tab20', 'tab20b'.
        - vertical_exaggeration: int, vertical exaggeration factor (default: 500 for 1:500 ratio)
        """
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Identify the depth column
        depth_column_aliases = ['Depth', 'DEPTH', 'depth']
        depth_column = next((alias for alias in depth_column_aliases if alias in df.columns), None)
        if not depth_column:
            raise ValueError("The CSV file must contain a depth column with one of the following aliases: 'Depth', 'DEPTH', or 'depth'.")

        # Sort the DataFrame by depth
        df = df.sort_values(by=depth_column, ascending=True)

        # Extract depth and logs
        depth = df[depth_column]
        logs = {col: df[col] for col in df.columns if col not in [depth_column, 'LITHOLOGY']}
        lithology_data = df[[depth_column, 'LITHOLOGY']] if 'LITHOLOGY' in df.columns else None

        # Calculate depth range for vertical exaggeration
        depth_range = depth.max() - depth.min()
        
        # Calculate figure height based on vertical exaggeration
        # For a 1:500 ratio, we need to make the figure height proportionally larger
        base_height = 14  # default height in inches
        if figure_height is None:
            # Calculate height based on depth range and vertical exaggeration
            # The magic number 0.05 is a scaling factor to convert from depth units to inches at the specified exaggeration
            figure_height = depth_range * 0.05 * (vertical_exaggeration / 100)
            
            # Enforce reasonable limits (min 14 inches, max 60 inches)
            figure_height = max(base_height, min(figure_height, 60))
            
            print(f"Calculated figure height: {figure_height} inches for vertical exaggeration 1:{vertical_exaggeration}")

        # Handle negative values in logs
        for log_name, log_data in logs.items():
            negative_indices = log_data < 0
            if any(negative_indices):
                print(f"Warning: {negative_indices.sum()} negative values found in {log_name}. Replacing with zero.")
                logs[log_name] = log_data.clip(lower=0)

        # Default log limits if none is provided
        if log_limits is None:
            log_limits = {
                "GR": (0, 200),        # API
                "RT": (0.2, 2000),     # ohm.m (log scale)
                "NPHI": (0.6, -0.05),  # v/v or p.u.
                "RHOB": (1.0, 3.0),    # g/cc
                "PEF": (0, 10),        # b/e
                "DT": (140, 40),       # us/ft
                "BVW": (0, 1),         # v/v
                "VSH": (0, 1),         # v/v
                "KLOGH": (0.001, 10000)  # mD (log scale)
            }

        # Add default limits for any logs not specified
        for log_name in logs.keys():
            if log_name not in log_limits:
                min_val = logs[log_name].min()
                max_val = logs[log_name].max()
                # Add a 10% buffer
                buffer = (max_val - min_val) * 0.1
                log_limits[log_name] = (min_val - buffer, max_val + buffer)

        # Handle track configuration
        if track_config is None:
            # Default categorization logic for logs
            track_config = self._auto_categorize_logs(logs)
        else:
            # Filter out logs that don't exist in the data
            track_config = {track: [log for log in track_logs if log in logs] 
                        for track, track_logs in track_config.items()}
            
            # Remove empty tracks
            track_config = {track: track_logs for track, track_logs in track_config.items() if track_logs}
            
            # Find uncategorized logs and create new tracks for them
            categorized_logs = [log for track_logs in track_config.values() for log in track_logs]
            uncategorized_logs = [log for log in logs.keys() if log not in categorized_logs]
            
            # Add new tracks for uncategorized logs
            if uncategorized_logs:
                track_config = self._add_uncategorized_logs(track_config, uncategorized_logs)

        # Determine the number of tracks
        num_tracks = len(track_config) + (1 if lithology_data is not None else 0)
        
        # ------ Dynamic Layout Optimization ------
        # Calculate optimal figure dimensions based on number of tracks
        if figure_height is None:
            figure_height = 14  # Default height
        
        # Dynamic width calculation based on number of tracks
        if figure_width is None:
            base_width_per_track = 3.5  # Base width per track in inches
            
            # Adjust width per track based on number of tracks (smaller width for many tracks)
            if num_tracks > 10:
                base_width_per_track = 2.8
            elif num_tracks > 6:
                base_width_per_track = 3.0
            
            # Calculate total width with a minimum of 10 inches
            figure_width = max(10, base_width_per_track * num_tracks)
        
        # ------ Color Scheme Selection ------
        # Set color scheme based on input or number of logs
        if color_scheme is None:
            total_logs = sum(len(logs_list) for logs_list in track_config.values())
            if total_logs > 20:
                color_scheme = 'tab20'
            elif total_logs > 10:
                color_scheme = 'tab10'
            else:
                color_scheme = 'default'
        
        # Get color map based on scheme
        if color_scheme == 'default':
            colors = plt.cm.tab10.colors
        elif color_scheme == 'viridis':
            cmap = plt.cm.viridis
            colors = [cmap(i) for i in np.linspace(0, 1, len(logs))]
        elif color_scheme == 'plasma':
            cmap = plt.cm.plasma
            colors = [cmap(i) for i in np.linspace(0, 1, len(logs))]
        elif color_scheme == 'tab20':
            colors = plt.cm.tab20.colors + plt.cm.tab20b.colors
        elif color_scheme == 'tab20b':
            colors = plt.cm.tab20b.colors + plt.cm.tab20.colors
        else:
            colors = plt.cm.tab10.colors
        
        # Generate unique colors for logs - using a better mapping approach
        all_logs = list(logs.keys())
        color_map = {log_name: colors[i % len(colors)] for i, log_name in enumerate(all_logs)}

        # Create subplots with improved spacing
        fig, axes = plt.subplots(nrows=1, ncols=num_tracks, figsize=(figure_width, figure_height), sharey=True,
                                gridspec_kw={'wspace': 0.1})  # Reduced space between subplots
        
        if num_tracks == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot

        # Plot logs in each track
        track_idx = 0
        for track_name, track_logs in track_config.items():
            ax = axes[track_idx]
            track_idx += 1
            
            # Calculate optimal number of logs per track
            max_logs_per_track = 4  # Default max logs per track
            if len(track_logs) > max_logs_per_track:
                print(f"Warning: Track '{track_name}' has {len(track_logs)} logs. Limiting to {max_logs_per_track} for clarity.")
                track_logs = track_logs[:max_logs_per_track]
            
            # If track has multiple logs, create additional twin axes
            twin_axes = []
            for i in range(1, len(track_logs)):
                twin_ax = ax.twiny()
                twin_axes.append(twin_ax)
                # Move twin axes spines to the right side incrementally
                twin_ax.spines['top'].set_position(('outward', 40 * i))
            
            # Plot first log on the primary axis
            if track_logs:
                first_log = track_logs[0]
                ax.plot(logs[first_log], depth, label=first_log, color=color_map[first_log], linewidth=1.5)
                ax.set_xlim(log_limits[first_log])
                ax.set_xlabel(f"{first_log}\n{log_limits[first_log][0]:.2f}-{log_limits[first_log][1]:.2f}", fontsize=9)
                
                # Fill between line and axis for the first log to enhance visibility
                ax.fill_betweenx(depth, ax.get_xlim()[0], logs[first_log], alpha=0.05, color=color_map[first_log])
            
            # Plot additional logs on twin axes
            for i, log_name in enumerate(track_logs[1:], 0):
                twin_ax = twin_axes[i]
                twin_ax.plot(logs[log_name], depth, label=log_name, color=color_map[log_name], linewidth=1.5)
                twin_ax.set_xlim(log_limits[log_name])
                twin_ax.set_xlabel(f"{log_name}\n{log_limits[log_name][0]:.2f}-{log_limits[log_name][1]:.2f}", fontsize=9)
                
                # Configure twin axis
                twin_ax.xaxis.set_ticks_position("top")
                twin_ax.xaxis.set_label_position("top")
                twin_ax.tick_params(axis='x', colors=color_map[log_name], labelsize=8)
                twin_ax.xaxis.label.set_color(color_map[log_name])
                
                # Optional: Add some transparency to the line
                twin_ax.get_lines()[0].set_alpha(0.8)
            
            # Configure primary axis
            if track_logs:
                ax.tick_params(axis='x', colors=color_map[track_logs[0]], labelsize=8)
                ax.xaxis.label.set_color(color_map[track_logs[0]])
            
            # Improved track title with number of logs indication
            ax.set_title(f"{track_name}\n({len(track_logs)} logs)", fontsize=10)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.1)  # Lighter grid
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            
            # Set y-tick labels only on first axis
            if track_idx > 1:
                ax.tick_params(axis='y', labelsize=8, labelleft=False)
            else:
                ax.tick_params(axis='y', labelsize=8)
            
            # Create a custom legend for the track with improved visibility
            handles = []
            for i, log_name in enumerate(track_logs):
                if i == 0:
                    handles.append(ax.get_lines()[0])
                else:
                    handles.append(twin_axes[i-1].get_lines()[0])
            
            # Place legend at bottom with enhanced styling
            leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), 
                        fontsize=12, frameon=True, fancybox=True, framealpha=0.2, draggable=True,
                        ncol=min(2, len(track_logs)))
            leg.get_frame().set_linewidth(0.5)

        # Plot lithology if available
        if lithology_data is not None:
            if depth_column not in lithology_data.columns or 'LITHOLOGY' not in lithology_data.columns:
                raise ValueError("The lithology_data must contain both the depth column and 'LITHOLOGY' column.")
            ax = axes[-1]
            depth_for_lithology = lithology_data[depth_column].values
            lithology = lithology_data['LITHOLOGY'].values
            lithology_dict = self.lithology_labels

            for i in range(len(depth_for_lithology) - 1):
                lith = lithology[i]
                color = lithology_dict.get(lith, {}).get('color', '#D2B48C')  # Default color
                hatch = lithology_dict.get(lith, {}).get('hatch', '')  # Default hatch
                ax.fill_betweenx([depth_for_lithology[i], depth_for_lithology[i + 1]], 0.5, 1, facecolor=color, alpha=0.5, hatch=hatch, linewidth=0.3)

            # Add lithology legend with improved layout and larger box size
            handles = [
                Patch(facecolor=attrs['color'], hatch=attrs['hatch'], edgecolor='k', label=lith)
                for lith, attrs in self.lithology_labels.items()
            ]
            #leg = ax.legend(handles=handles, title='Lithology', bbox_to_anchor=(0.5, 1), loc='upper left', fontsize=10)
            # Adjust legend spacing and layout
            #leg._legend_box.align = "upper center"  # Align legend items to the center
            
            # Use multiple columns for the legend if needed
            ncols = max(1, min(3, len(handles) // 4 + 1))
            leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                        ncol=ncols, fontsize=10, frameon=True, fancybox=True)
            leg.get_frame().set_linewidth(1.5)
            
            ax.set_title('Lithology', fontsize=12)
            ax.set_xticks([])  # Hide x-axis for lithology
            ax.invert_yaxis()

        # Set common properties
        for ax in axes:
            ax.set_ylim(depth.max(), depth.min())  # Invert depth axis
            # Add minor grid lines for better readability
            ax.grid(which='minor', alpha=0.15)
            ax.minorticks_on()
        
        # Bold the depth axis label and make it larger
        axes[0].set_ylabel("Depth (m)", fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='y', labelsize=9)

        # Add log scales for certain logs
        for track_name, track_logs in track_config.items():
            ax_idx = list(track_config.keys()).index(track_name)
            ax = axes[ax_idx]
            
            # Apply log scale for specific logs (RT and KLOGH)
            log_scale_logs = ['RT', 'KLOGH', 'RD', 'RS', 'RESD', 'RESS', 'ILD', 'ILM', 'LLD', 'LLM', 'RES']
            
            if track_logs and track_logs[0] in log_scale_logs:
                ax.set_xscale('log')
                # For log scales, ensure we have reasonable tick marks
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
                
            # Apply log scale for twin axes if needed
            for i, log_name in enumerate(track_logs[1:], 0):
                if log_name in log_scale_logs and i < len(twin_axes):
                    twin_axes[i].set_xscale('log')
                    twin_axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

        # Add depth markers every 100m or appropriate interval
        depth_range = depth.max() - depth.min()
        if vertical_exaggeration >= 500:
            # For high vertical exaggeration, use smaller intervals
            if depth_range > 1000:
                interval = 50  # Every 50m for large intervals with high exaggeration
            elif depth_range > 500:
                interval = 25  # Every 25m for medium intervals with high exaggeration
            elif depth_range > 100:
                interval = 10  # Every 10m for smaller intervals with high exaggeration
            else:
                interval = 5   # Every 5m for very small intervals with high exaggeration
        else: 
            if depth_range > 1000:
                interval = 200  # Every 200m for large intervals
            elif depth_range > 500:
                interval = 100  # Every 100m for medium intervals
            elif depth_range > 100:
                interval = 50   # Every 50m for smaller intervals
            else:
                interval = 10   # Every 10m for very small intervals
        
        # Calculate depth markers
        start_depth = math.ceil(depth.min() / interval) * interval
        end_depth = math.floor(depth.max() / interval) * interval
        depth_markers = np.arange(start_depth, end_depth + interval, interval)
        
        # Add reference lines at depth markers across all tracks
        for d in depth_markers:
            for ax in axes:
                ax.axhline(y=d, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)

        # Improved title with well metadata
        title_text = f"Well Logs and Lithology: {well_name}"
        subtitle_text = f"Depth Range: {depth.min():.1f}m - {depth.max():.1f}m | Total Logs: {len(logs)}"
        
        # Dynamically adjust title placement based on layout
        if len(axes) > 1:
            fig.suptitle(title_text, fontsize=24, y=1.03, fontweight='bold')
            fig.text(0.5, 0.98, subtitle_text, ha='center', fontsize=16)
        else:
            fig.suptitle(title_text, fontsize=24, y=1.03, fontweight='bold')
            fig.text(1.0, 0.98, subtitle_text, ha='center', fontsize=16)
        
        # Add a timestamp to the plot
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Automatically adjust position based on master layout
        if len(axes) > 1:
            fig.text(0.9, 0.045, f"Generated: {timestamp}", fontsize=16, color='gray', ha='right')
        else:
            fig.text(1.0, 0.045, f"Generated: {timestamp}", fontsize=16, color='gray', ha='right')
        
        # Make sure the plot uses the full figure area. Adjust as necessary "left, bottom, right, top"
        try:
            plt.tight_layout(rect=[0, 0, 0.5, 0.95])  # Master layout for main log plot
        except ValueError:
            print("Warning: Tight layout adjustment failed. Using default layout.")
        
        # Add bottom text showing log information
        log_info_text = f"Log statistics: Mean depth increment: {np.mean(np.diff(depth)):.2f}m | Data points: {len(depth)}"
        
        # Automatically adjust position based on master layout
        if len(axes) > 1:
            fig.text(0.5, 0.045, log_info_text, ha='center', fontsize=16, style='italic')
        else:
            fig.text(1.0, 0.045, log_info_text, ha='center', fontsize=16, style='italic')
        
        plt.show()
        
        return fig, axes  # Return the figure and axes for further customization if needed

    def _optimize_track_layout(self, logs, track_config, max_logs_per_track=4):
        """
        Optimizes track layout based on the number of logs and their correlation.
        
        Parameters:
        - logs: dict, dictionary of log names and their data
        - track_config: dict, track configuration with logs categorized
        - max_logs_per_track: int, maximum number of logs per track for readability
        
        Returns:
        - optimized_track_config: dict, optimized track configuration
        """
        optimized_track_config = {}
        
        # Process each track
        for track_name, track_logs in track_config.items():
            # If track has too many logs, split it
            if len(track_logs) > max_logs_per_track:
                # Calculate correlations between logs in this track
                log_data = pd.DataFrame({log: logs[log] for log in track_logs if log in logs})
                
                # If we can calculate correlations, use them to group similar logs
                if len(log_data.columns) > 1:
                    corr_matrix = log_data.corr().abs()
                    
                    # Create clusters of logs based on correlation
                    groups = []
                    remaining_logs = set(track_logs)
                    
                    while remaining_logs:
                        # Start a new group with the first remaining log
                        current_log = next(iter(remaining_logs))
                        current_group = [current_log]
                        remaining_logs.remove(current_log)
                        
                        # Find up to (max_logs_per_track-1) more logs that correlate well
                        if current_log in corr_matrix.index:
                            correlated_logs = corr_matrix[current_log].sort_values(ascending=False).index.tolist()
                            for log in correlated_logs:
                                if log in remaining_logs and len(current_group) < max_logs_per_track:
                                    current_group.append(log)
                                    remaining_logs.remove(log)
                        
                        groups.append(current_group)
                    
                    # Create new tracks for each group
                    base_name = track_name.split('(')[0].strip()
                    for i, group in enumerate(groups):
                        optimized_track_config[f"{base_name} {i+1} ({track_name.split('(')[1]}"] = group
                else:
                    # If correlation can't be used, split logs evenly
                    chunks = [track_logs[i:i + max_logs_per_track] 
                            for i in range(0, len(track_logs), max_logs_per_track)]
                    
                    for i, chunk in enumerate(chunks):
                        optimized_track_config[f"{track_name} {i+1}"] = chunk
            else:
                # Keep tracks with acceptable number of logs as they are
                optimized_track_config[track_name] = track_logs
        
        return optimized_track_config

    def _auto_categorize_logs(self, logs):
        """
        Automatically categorize logs into appropriate tracks based on their types.
        Enhanced with better categorization and handling of specialized logs.
        
        Parameters:
        - logs: dict, dictionary of log names and their data
        
        Returns:
        - track_config: dict, track configuration with logs categorized
        """
        # Define log categories with common logs in each category
        log_categories = {
            "Resistivity": ["RT", "RD", "RS", "RESD", "RESS", "RESM", "RXO", "Rxo", "ILD", "ILM", "LLD", "LLM", "RES"],
            "Density": ["RHOB", "DEN", "RHOZ", "RHO", "Rhoma", "DENS"],
            "Sonic": ["DT", "DTS", "DTP", "DTCO", "DTSM", "DTPM", "SONIC", "Sonic", "DTC", "DTSF"],
            "Velocity": ["Vp", "Vs"],
            "Porosity": ["NPHI", "DPHI", "SPHI", "PHIF", "PHI", "PHIE", "PHIT", "PHIA", "POR"],
            "Gamma Ray": ["GR", "CGR", "SGR", "Gamma", "NGR"],
            "Calculated logs": ["VSH", "BVW", "SW", "Sxo", "Swi", "Sro", "Swc", "VCL", "VCL_GR"],
            "Permeability": ["KLOGH", "PERM", "Kx", "Ky", "Kz", "PERMY", "PERMZ", "PERMX"],
            "Photoelectric Factor": ["PEF", "PE", "UMA"],
            "Caliper": ["CALI", "CALD", "CAL", "CALX", "CALY", "HCAL"],
            # Elements by IUPAC nomenclature
            "Actinides": ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"],
            "Lanthanides": ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
            "Noble gases": ["He", "Ne", "Ar", "Kr", "Xe", "Rn"],
            "Non-metals": ["H", "C", "N", "O", "P", "S", "Se"],
            "Halogens": ["F", "Cl", "Br", "I"],
            "Metalloids": ["B", "Si", "Ge", "As", "Sb", "Te", "At"],
            "Alkaline earth metals": ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],
            "Alkali metals": ["Li", "Na", "K", "Rb", "Cs", "Fr"],
            "Transition metals1": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
            "Transition metals2": ["Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"],
            "Transition metals3": ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"],
            "Post-transition metals": ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Po"],
            "Trace Elements1": ["Zn", "U", "As", "Ba", "Sr", "Zr", "Ni", "Cu", "Co"],
            "Trace Elements2": ["Pb", "Cr", "V", "Mo", "Ag", "Hg", "Cd", "Sb", "Bi"],
            "Major Oxides1": ["SiO2", "Al2O3", "CaO", "MgO", "Na2O", "K2O", "TiO2",],
            "Major Oxides2": ["P2O5", "SO3", "MnO", "Fe2O3", "FeO", "H2O", "CO2"],
            "Minor Oxides1": ["ZnO", "UO2", "As2O3", "BaO", "SrO", "ZrO2", "NiO", "CuO", "CoO"],
            "Minor Oxides2": ["PbO", "Cr2O3", "V2O5", "MoO3", "Ag2O", "HgO", "CdO", "Sb2O3", "Bi2O3"],
            "Heavy Metals": ["Pb", "Hg", "Cd", "As", "Cr", "Ni", "Cu", "Zn", "Co", "V", "Mo", "Sb", "Bi", "Th", "U"],
            "Rare Earth Elements": ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
            
            # Organic geochemical measurements
            "Organic Geochemistry": ["TOC", "CEC", "CHNS", "HI", "OI", "PI", "S1", "S2", "S3", "Tmax"],
            "Fluid Saturation": ["SW", "SG", "SO", "SXO", "SWI", "SOR"],
            # Spectroscopy logs
            "Spectroscopy": ["LE", "ECS", "XRD", "XRF", "NGS", "HNGS", "SPEC", "ELAN"],
            # Nuclear logs
            "Nuclear": ["TNPH", "CNL", "SNP", "FDC", "LDT", "GNT"],
            # Drilling and formation parameters
            "Drilling": ["ROP", "RPM", "WOB", "TRQ", "SPP", "ECD"],
            # Formation testing
            "Formation Testing": ["MDT", "RFT", "FIT", "LOT", "PRES", "TEMP"],
            # Electrical logs
            "Electrical": ["SP", "MSFL", "ML", "SFL"]
        }
        
        # Initialize track configuration
        track_config = {}
        
        # Function to find the category for a log
        def find_category(log_name):
            # First try exact matches
            for category, log_list in log_categories.items():
                if log_name in log_list:
                    return category
                    
            # Then try substring matches
            for category, log_list in log_categories.items():
                # Check if the log name contains any of the category's patterns
                if any(log.upper() in log_name.upper() for log in log_list):
                    return category
                    
            # Try to match by common prefixes/suffixes
            prefixes = {
                "DT": "Sonic",
                "GR": "Gamma Ray", 
                "RH": "Density",
                "PH": "Porosity",
                "SW": "Fluid Saturation",
                "PE": "Photoelectric Factor",
                "CA": "Caliper"
            }
            
            for prefix, category in prefixes.items():
                if log_name.upper().startswith(prefix):
                    return category
                    
            return "Other"
        
        # Categorize logs
        categorized_logs = {}
        for log_name in logs.keys():
            category = find_category(log_name)
            if category not in categorized_logs:
                categorized_logs[category] = []
            categorized_logs[category].append(log_name)
        
        # Create tracks based on categories
        track_num = 1
        for category, log_list in categorized_logs.items():
            if category == "Other":
                # For "Other" category, create one track per log or small groups
                if len(log_list) <= 4:
                    track_config[f"Track {track_num} (Other)"] = log_list
                    track_num += 1
                else:
                    # Split into chunks of at most 4 logs
                    chunks = [log_list[i:i + 4] for i in range(0, len(log_list), 4)]
                    for i, chunk in enumerate(chunks):
                        track_config[f"Track {track_num} (Other {i+1})"] = chunk
                        track_num += 1
            else:
                # If category has many logs, split it into multiple tracks
                if len(log_list) > 4:
                    chunks = [log_list[i:i + 4] for i in range(0, len(log_list), 4)]
                    for i, chunk in enumerate(chunks):
                        track_config[f"Track {track_num} ({category} {i+1})"] = chunk
                        track_num += 1
                else:
                    # Group logs by their category
                    track_config[f"Track {track_num} ({category})"] = log_list
                    track_num += 1
        
        return track_config

    def _add_uncategorized_logs(self, track_config, uncategorized_logs):
        """
        Add uncategorized logs to new tracks with smart grouping.
        
        Parameters:
        - track_config: dict, existing track configuration
        - uncategorized_logs: list, logs that need to be added to new tracks
        
        Returns:
        - updated_track_config: dict, updated track configuration
        """
        # Find the highest track number
        track_numbers = []
        for track_name in track_config.keys():
            if "Track" in track_name and "(" in track_name:
                try:
                    track_num = int(track_name.split("Track ")[1].split(" (")[0])
                    track_numbers.append(track_num)
                except (ValueError, IndexError):
                    continue
        
        next_track_num = max(track_numbers) + 1 if track_numbers else 1
        
        # Add new tracks for uncategorized logs

    def _add_uncategorized_logs(self, track_config, uncategorized_logs):
        """
        Add uncategorized logs to new tracks with smart grouping.
        
        Parameters:
        - track_config: dict, existing track configuration
        - uncategorized_logs: list, logs that need to be added to new tracks
        
        Returns:
        - updated_track_config: dict, updated track configuration
        """
        # Find the highest track number
        track_numbers = []
        for track_name in track_config.keys():
            if "Track" in track_name and "(" in track_name:
                try:
                    track_num = int(track_name.split("Track ")[1].split(" (")[0])
                    track_numbers.append(track_num)
                except (ValueError, IndexError):
                    continue
        
        next_track_num = max(track_numbers) + 1 if track_numbers else 1
        
        # Smart grouping for uncategorized logs
        if len(uncategorized_logs) <= 4:
            # If we have 4 or fewer logs, put them in a single track
            track_config[f"Track {next_track_num} (Uncategorized)"] = uncategorized_logs
        else:
            # Group similar logs together based on naming patterns and statistical analysis
            log_groups = {}
            
            # First pass: group by base name prefix
            for log in uncategorized_logs:
                # Extract base name prefix (first 2-3 characters, often indicating log type)
                prefix = ''.join([c for c in log[:3] if c.isalpha()]).upper()
                if prefix and len(prefix) >= 2:
                    if prefix not in log_groups:
                        log_groups[prefix] = []
                    log_groups[prefix].append(log)
            
            # Second pass: handle remaining logs
            ungrouped_logs = [log for log in uncategorized_logs if not any(log in group for group in log_groups.values())]
            if ungrouped_logs:
                for log in ungrouped_logs:
                    # Try to find best match among existing groups
                    best_match = None
                    highest_similarity = 0
                    
                    for prefix, group in log_groups.items():
                        # Simple similarity measure - length of common substring
                        similarity = len(set(log.upper()) & set(prefix))
                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            best_match = prefix
                    
                    if best_match and highest_similarity >= 1:
                        log_groups[best_match].append(log)
                    else:
                        # Create a new group for this log
                        new_group = log[:2].upper()
                        if new_group not in log_groups:
                            log_groups[new_group] = []
                        log_groups[new_group].append(log)
            
            # Create new tracks for each group, ensuring max 4 logs per track
            for prefix, logs_in_group in log_groups.items():
                if len(logs_in_group) <= 4:
                    track_config[f"Track {next_track_num} ({prefix})"] = logs_in_group
                    next_track_num += 1
                else:
                    # Split into chunks of at most 4 logs
                    chunks = [logs_in_group[i:i+4] for i in range(0, len(logs_in_group), 4)]
                    for i, chunk in enumerate(chunks):
                        track_config[f"Track {next_track_num} ({prefix} {i+1})"] = chunk
                        next_track_num += 1
        
        return track_config

    def generate_advanced_visualization(self, fig, axes, well_name, depth, logs, track_config, lithology_data=None):
        """
        Enhances the well log visualization with additional features like crossplots,
        statistical summaries, and zone identification.
        
        Parameters:
        - fig: matplotlib figure object from the main plot
        - axes: matplotlib axes from the main plot
        - well_name: str, name of the well
        - depth: pandas Series, depth data
        - logs: dict, dictionary of log data
        - track_config: dict, track configuration
        - lithology_data: pandas DataFrame, optional lithology data
        
        Returns:
        - fig: Enhanced matplotlib figure
        """
        import matplotlib.gridspec as gridspec
        import warnings
        warnings.filterwarnings("ignore")  # Suppress matplotlib warnings
        
        # Create a new figure for advanced visualization
        fig_advanced = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig_advanced, height_ratios=[1, 1])
        
        # 1. Generate crossplot between key log pairs 
        # Common crossplot pairs: (NPHI, RHOB), (GR, RT), (DT, RHOB)
        crossplot_pairs = [
            ("NPHI", "RHOB", "Neutron-Density Crossplot"),
            ("GR", "RT", "GR-Resistivity Crossplot"),
            ("DT", "RHOB", "Sonic-Density Crossplot")
        ]
        
        # Find available crossplot pairs
        available_pairs = []
        for x_log, y_log, title in crossplot_pairs:
            if x_log in logs and y_log in logs:
                available_pairs.append((x_log, y_log, title))
        
        # Plot up to 3 crossplots
        for i, (x_log, y_log, title) in enumerate(available_pairs[:3]):
            ax = fig_advanced.add_subplot(gs[0, i])
            x_data = logs[x_log]
            y_data = logs[y_log]
            
            # Create crossplot with color based on depth
            sc = ax.scatter(x_data, y_data, c=depth, cmap='viridis', 
                        alpha=0.6, edgecolors='none', s=15)
            
            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Depth (m)')
            
            # Add labels and title
            ax.set_xlabel(x_log)
            ax.set_ylabel(y_log)
            ax.set_title(title)
            
            # Add gridlines
            ax.grid(True, alpha=0.3)
            
            # If neutron-density crossplot, add sandstone and limestone lines
            if x_log == "NPHI" and y_log == "RHOB":
                # Sandstone line: approximate neutron-density relationship
                neutron_values = np.linspace(0, 0.5, 100)
                # Approximate limestone line
                limestone_density = 2.71 - 1.0 * neutron_values
                ax.plot(neutron_values, limestone_density, 'r--', label='Limestone')
                
                # Approximate sandstone line 
                sandstone_density = 2.65 - 0.8 * neutron_values
                ax.plot(neutron_values, sandstone_density, 'g--', label='Sandstone')
                
                # Approximate dolomite line
                dolomite_density = 2.87 - 1.2 * neutron_values
                ax.plot(neutron_values, dolomite_density, 'b--', label='Dolomite')
                
                ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True)
        
        # 2. Statistical Summary / Log Distributions
        ax_stats = fig_advanced.add_subplot(gs[1, 0])
        
        # Select important logs for statistics
        key_logs = []
        priority_logs = ["GR", "RT", "NPHI", "RHOB", "DT", "SW", "VSH", "PERM"]
        
        # Add logs based on priority
        for log_name in priority_logs:
            if log_name in logs:
                key_logs.append(log_name)
        
        # If we don't have at least 4 logs, add some others
        if len(key_logs) < 4:
            for log_name in logs:
                if log_name not in key_logs:
                    key_logs.append(log_name)
                    if len(key_logs) >= 4:
                        break
        
        # Calculate statistics for key logs
        stats_data = []
        for log_name in key_logs:
            log_data = logs[log_name]
            stats = {
                'Log': log_name,
                'Mean': np.mean(log_data),
                'Min': np.min(log_data),
                'Max': np.max(log_data),
                'StdDev': np.std(log_data),
                'P10': np.percentile(log_data, 10),
                'P50': np.percentile(log_data, 50),
                'P90': np.percentile(log_data, 90)
            }
            stats_data.append(stats)
        
        # Create a table for statistics
        cell_text = []
        for stat in stats_data:
            cell_text.append([
                f"{stat['Log']}", 
                f"{stat['Mean']:.2f}", 
                f"{stat['Min']:.2f}", 
                f"{stat['Max']:.2f}",
                f"{stat['StdDev']:.2f}",
                f"{stat['P50']:.2f}"
            ])
        
        # Hide axes
        ax_stats.axis('off')
        
        # Add table
        the_table = ax_stats.table(
            cellText=cell_text,
            colLabels=['Log', 'Mean', 'Min', 'Max', 'StdDev', 'Median'],
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1.2, 1.5)
        for i, key in enumerate(the_table._cells):
            cell = the_table._cells[key]
            if i < 6:  # Header row
                cell.set_facecolor('#d8e8f0')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#f2f2f2')
        
        ax_stats.set_title("Statistical Summary", fontsize=12)
        
        # 3. Histogram of key petrophysical parameter
        ax_hist = fig_advanced.add_subplot(gs[1, 1])
        
        # Choose the most relevant log for histogram (GR, NPHI, RHOB or VSH)
        histogram_log_candidates = ["GR", "NPHI", "RHOB", "VSH"]
        histogram_log = None
        
        for log_name in histogram_log_candidates:
            if log_name in logs:
                histogram_log = log_name
                break
        
        if histogram_log:
            # Create histogram
            data = logs[histogram_log]
            ax_hist.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax_hist.set_xlabel(histogram_log)
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title(f'{histogram_log} Distribution')
            ax_hist.grid(True, alpha=0.3)
            
            # Add mean and median lines
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax_hist.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax_hist.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
            ax_hist.legend()
        
        # 4. Correlation matrix heatmap
        ax_corr = fig_advanced.add_subplot(gs[1, 2])
        
        # Choose logs for correlation matrix
        corr_logs = []
        for log_name in logs:
            # Skip logs with too many missing values
            if np.isnan(logs[log_name]).sum() < len(logs[log_name]) * 0.1:
                corr_logs.append(log_name)
                if len(corr_logs) >= 8:  # Limit to 8 logs for readability
                    break
        
        if corr_logs:
            # Create dataframe for correlation
            corr_data = pd.DataFrame({log: logs[log] for log in corr_logs})
            
            # Calculate correlation matrix
            corr_matrix = corr_data.corr()
            
            # Create heatmap
            im = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add colorbar
            plt.colorbar(im, ax=ax_corr)
            
            # Add labels
            tick_marks = np.arange(len(corr_logs))
            ax_corr.set_xticks(tick_marks)
            ax_corr.set_yticks(tick_marks)
            ax_corr.set_xticklabels(corr_logs, rotation=45, ha='right')
            ax_corr.set_yticklabels(corr_logs)
            
            # Add title
            ax_corr.set_title('Log Correlation Matrix')
            
            # Add correlation values
            for i in range(len(corr_logs)):
                for j in range(len(corr_logs)):
                    text = ax_corr.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                    ha="center", va="center", color="black", fontsize=8)
        
        # Title for the entire figure
        fig_advanced.suptitle(f"Advanced Analysis - {well_name}", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig_advanced

    def enhance_plot_appearance(self, fig, axes, track_config):
        """
        Enhances the visual appearance of the well log plot.
        
        Parameters:
        - fig: matplotlib figure object
        - axes: matplotlib axes objects
        - track_config: dict, track configuration
        
        Returns:
        - fig: Enhanced figure
        - axes: Enhanced axes
        """
        # Improve overall visual style
        plt.style.use('seaborn-v0_8-whitegrid')  # Use a cleaner style
        
        # Add subtle background grid patterns to alternate tracks for visual separation
        for i, ax in enumerate(axes):
            if i % 2 == 0:
                ax.set_facecolor('#f9f9f9')  # Light gray background for even tracks
        
        # Enhance title and axis labels
        for i, (track_name, track_logs) in enumerate(track_config.items()):
            if i < len(axes):
                # Improved track title formatting
                axes[i].set_title(track_name, fontsize=10, fontweight='bold', pad=10)
                
                # Enhance axis label visibility
                if i == 0:  # First track
                    axes[i].set_ylabel("Depth (m)", fontsize=12, fontweight='bold')
        
        # Add frame around the entire figure
        fig.patch.set_linewidth(2)
        fig.patch.set_edgecolor('black')
        
        # Add subtle rounded corners to the figure
        for ax in axes:
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        
        # Make depth axis more prominent
        if len(axes) > 0:
            axes[0].spines['left'].set_linewidth(1.5)
            axes[0].spines['left'].set_color('black')
        
        # Customize tick parameters
        for ax in axes:
            ax.tick_params(axis='both', which='major', direction='in', length=4, width=1, pad=3)
            ax.tick_params(axis='both', which='minor', direction='in', length=2, width=0.5)
        
        return fig, axes

    def prepare_well_log_image_export(self, fig, output_path=None, dpi=300, format='png', add_border=True):
        """
        Prepares the well log visualization for high-quality export.
        
        Parameters:
        - fig: matplotlib figure object
        - output_path: str, optional path to save the figure
        - dpi: int, resolution of the output image
        - format: str, file format ('png', 'jpg', 'pdf', 'svg')
        - add_border: bool, whether to add a border to the image
        
        Returns:
        - str: Path to the saved image or None if not saved
        """
        if add_border:
            # Add a border to the figure
            fig.patch.set_linewidth(2)
            fig.patch.set_edgecolor('black')
            fig.patch.set_facecolor('white')
        
        # Set figure size to ensure output quality
        fig.set_size_inches(fig.get_size_inches())
        
        # Save figure if output path is provided
        if output_path:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save with high quality
            fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight', pad_inches=0.1)
            print(f"Figure saved to {output_path}")
            return output_path
        
        return None

    def plot_heatmap(self, csv_file_path, elements):
        """
        Plots a heatmap for the specified elements from the well data.

        Parameters:
        - file_path (str): Path to the well data CSV file.
        - elements (list): List of elements to include in the heatmap.

        Returns:
        - None
        """

        # Load the well data
        try:
            data = pd.read_csv(csv_file_path)
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

        # Check if all elements are present in the data
        missing_elements = [el for el in elements if el not in data.columns]
        if missing_elements:
            raise ValueError(f"Missing elements in data: {missing_elements}")

        # Filter the data for the specified elements
        heatmap_data = data[elements]

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Heatmap of Element Correlations")
        plt.show()
        

# References
#Andy McDonald. (2020). Petrophysics-Python-Series/14 - Displaying Lithology Data.ipynb at master · andymcdgeo/Petrophysics-Python-Series. GitHub. https://github.com/andymcdgeo/Petrophysics-Python-Series/blob/master/14%20-%20Displaying%20Lithology%20Data.ipynb