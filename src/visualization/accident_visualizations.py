import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os

# Set the style for better visualizations
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Load the accident data
def load_data(file_path):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} accident records")
    return df

# Process date and time information
def process_datetime(df):
    # Process the date
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    
    # Extract day of week
    df['Day_of_Week'] = df['Date'].dt.day_name()
    
    # Process time - the format might be complex, so we'll handle it carefully
    # Example: "4/12/2019 7:45:00 PM"
    
    # First, we'll extract just the time portion with AM/PM
    df['Time_Processed'] = df['Time'].str.extract(r'(\d+:\d+:\d+ [AP]M)')[0]
    
    # Convert to datetime
    df['Time_Processed'] = pd.to_datetime(df['Time_Processed'], format='%I:%M:%S %p', errors='coerce')
    
    # Extract hour
    df['Hour'] = df['Time_Processed'].dt.hour
    
    return df

# Create Visualization 1: Accident Types Pie Chart
def plot_accident_types(df, save_path=None):
    plt.figure(figsize=(10, 8))
    
    # Count accident types
    accident_counts = df['Accident Type'].value_counts()
    
    # Only include significant types (more than 10 occurrences)
    significant_types = accident_counts[accident_counts > 10]
    
    # Create explode effect for better visualization
    explode = [0.1] + [0] * (len(significant_types) - 1)
    
    # Create pie chart
    plt.pie(significant_types, labels=significant_types.index, autopct='%1.1f%%', 
            startangle=90, explode=explode, shadow=True)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    plt.title('Accident Types Distribution', fontsize=16)
    
    if save_path:
        plt.savefig(f"{save_path}/accident_types_pie.png", dpi=300, bbox_inches='tight')
    
    plt.show()

# Create Visualization 2: Accidents by Hour of Day
def plot_accidents_by_hour(df, save_path=None):
    plt.figure(figsize=(12, 8))
    
    # Create the plot
    ax = sns.countplot(x='Hour', data=df, palette='viridis')
    
    # Add labels and title
    plt.xlabel('Hour of Day (24h)', fontsize=14)
    plt.ylabel('Number of Accidents', fontsize=14)
    plt.title('Accidents by Hour of Day', fontsize=16)
    
    # Add count labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom', 
                    fontsize=10)
    
    # Ensure all x-ticks are visible
    plt.xticks(range(0, 24))
    
    if save_path:
        plt.savefig(f"{save_path}/accidents_by_hour.png", dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

# Create Visualization 3: Accidents by Day of Week
def plot_accidents_by_day(df, save_path=None):
    plt.figure(figsize=(12, 8))
    
    # Create a custom order for days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Count accidents by day and reindex to ensure correct order
    day_counts = df['Day_of_Week'].value_counts().reindex(day_order)
    
    # Create the bar plot
    ax = sns.barplot(x=day_counts.index, y=day_counts.values, palette='cool')
    
    # Add labels and title
    plt.xlabel('Day of Week', fontsize=14)
    plt.ylabel('Number of Accidents', fontsize=14)
    plt.title('Accidents by Day of Week', fontsize=16)
    
    # Add count labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   fontsize=12)
    
    if save_path:
        plt.savefig(f"{save_path}/accidents_by_day.png", dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

# Create Visualization 4: Accident Severity
def plot_accident_severity(df, save_path=None):
    plt.figure(figsize=(10, 8))
    
    # Categorize accidents by severity
    severity_mapping = {
        'Property Damage Crash': 'Property Damage Only',
        'PD': 'Property Damage Only',
        'pd': 'Property Damage Only',
        'Injury Crash': 'Injury',
        'PI': 'Injury',
        'Fatal Crash': 'Fatal',
        'F': 'Fatal'
    }
    
    # Apply mapping and count
    df['Severity'] = df['Accident Type'].map(severity_mapping)
    severity_counts = df['Severity'].value_counts()
    
    # Create pie chart
    colors = ['#ff9999','#66b3ff','#99ff99']
    plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=colors, shadow=True)
    plt.axis('equal')
    plt.title('Accident Severity Distribution', fontsize=16)
    
    if save_path:
        plt.savefig(f"{save_path}/accident_severity.png", dpi=300, bbox_inches='tight')
    
    plt.show()

# Create a heatmap showing accident locations (if geo data is available)
def plot_accident_heatmap(df, save_path=None):
    try:
        # Extract coordinates from the georeferenced column
        # Assuming format like "(lat, long)"
        df['Coordinates'] = df['New Georeferenced Column'].str.extract(r'\(([^,]+),\s*([^)]+)\)')
        df['Latitude'] = pd.to_numeric(df['Coordinates'][0], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Coordinates'][1], errors='coerce')
        
        # Check if we have enough valid coordinates
        valid_coords = df.dropna(subset=['Latitude', 'Longitude'])
        if len(valid_coords) > 10:
            plt.figure(figsize=(12, 10))
            
            # Create a 2D histogram
            plt.hist2d(valid_coords['Longitude'], valid_coords['Latitude'], 
                      bins=50, cmap='hot')
            
            plt.colorbar(label='Number of Accidents')
            plt.title('Accident Location Heatmap', fontsize=16)
            plt.xlabel('Longitude', fontsize=14)
            plt.ylabel('Latitude', fontsize=14)
            
            if save_path:
                plt.savefig(f"{save_path}/accident_heatmap.png", dpi=300, bbox_inches='tight')
            
            plt.tight_layout()
            plt.show()
        else:
            print("Not enough valid coordinates for a heatmap")
    except Exception as e:
        print(f"Could not create heatmap: {e}")

# Main function to run all visualizations
def analyze_accidents(file_path, save_path=None):
    # Create the output directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    
    # Load and process data
    df = load_data(file_path)
    df = process_datetime(df)
    
    # Create all visualizations
    plot_accident_types(df, save_path)
    plot_accidents_by_hour(df, save_path)
    plot_accidents_by_day(df, save_path)
    plot_accident_severity(df, save_path)
    
    # Try to create a heatmap if geo data is available
    plot_accident_heatmap(df, save_path)
    
    # Print some key statistics for the report
    print("\n--- KEY STATISTICS ---")
    print(f"Total number of accidents: {len(df)}")
    print(f"Accidents by type:\n{df['Accident Type'].value_counts()}")
    print(f"\nPeak accident hour: {df['Hour'].value_counts().idxmax()}")
    print(f"Peak accident day: {df['Day_of_Week'].value_counts().idxmax()}")

# Run the analysis
if __name__ == "__main__":
    # Update these paths as needed
    data_file = "MDTA_Accidents_20250316.csv"
    
    analyze_accidents(data_file, None)