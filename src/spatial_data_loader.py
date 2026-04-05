import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
# import simple base map
import contextily as ctx
from shapely import wkt

from collections import defaultdict
from matplotlib.patches import Patch

import osmnx as ox
import networkx as nx
from shapely.geometry import Point

from tqdm import tqdm

def get_splited_road_index(df):
    # Slice the index to keep the first 4 characters
    index_split = df.index.str[:4]

    # Get unique values
    index_split = pd.unique(index_split)

    # Convert to a list
    index_split = index_split.tolist()
    return index_split


def get_road_sensors():
    """
    Load road sensors data from a CSV file.
    """
    # Load the road sensors data from a GeoPackage file
    road_sensors = gpd.read_file("data/raw/MIV/9c1cc8df-cc5c-11ef-8ebb-005056b0ce82/data/data.gpkg")
    
    # Only keep poi_id and geometry and convert it to a gpd with crs lv95
    road_sensors = road_sensors[['zsid', 'geometry']]
    road_sensors = road_sensors.rename(columns={"zsid": "sensor_id"})
    road_sensors = gpd.GeoDataFrame(road_sensors, geometry='geometry', crs="EPSG:2056")
    
    road_sensors_filtered = pd.read_csv("data/processed/MIV/traffic_all_years_filtered.csv", index_col=0)
    road_sensors_filtered = get_splited_road_index(road_sensors_filtered)
    
    # Add column for use in model
    road_sensors['model_use'] = False
    road_sensors.loc[road_sensors['sensor_id'].isin(road_sensors_filtered), 'model_use'] = True
    
    # Add column for mode of transport
    road_sensors['transport_mode'] = 'road'
    
    return road_sensors


def get_active_modes_sensors():
    """
    Load bike sensors data from a CSV file.
    """
    # Load the bike sensors data
    active_modes_sensors = pd.read_csv("data/raw/Langsamverkehr/e1f85837-cc5a-11ef-9763-005056b0ce82/data/data_points.csv")
    
    # Only keep the column id1 and geometry
    active_modes_sensors = active_modes_sensors[['id1', 'geometry']]
    active_modes_sensors = active_modes_sensors.rename(columns={"id1": "sensor_id"})
    active_modes_sensors['geometry'] = active_modes_sensors['geometry'].apply(wkt.loads)
    # Convert to a gpd with crs lv95
    active_modes_sensors = gpd.GeoDataFrame(active_modes_sensors, geometry='geometry', crs="EPSG:2056")
    # Add column for mode of transport
    
    # Load data from a CSV file
    bike_sensors = pd.read_csv("data/processed/Langsamverkehr/bike_all_years.csv")
    ped_sensor = pd.read_csv("data/processed/Langsamverkehr/pedestrian_all_years.csv")
    
    # Extract fk_standort to list
    bike_sensors = bike_sensors['FK_STANDORT'].unique()
    ped_sensor = ped_sensor['FK_STANDORT'].unique()
    
    # Add column for mode of transport
    active_modes_sensors['transport_mode'] = None
    active_modes_sensors.loc[active_modes_sensors['sensor_id'].isin(bike_sensors), 'transport_mode'] = 'bike'
    active_modes_sensors.loc[active_modes_sensors['sensor_id'].isin(ped_sensor), 'transport_mode'] = 'pedestrian'
    
    filtered_bike_sensors = pd.read_csv("data/processed/Langsamverkehr/bike_all_years_filtered.csv")
    filtered_ped_sensors = pd.read_csv("data/processed/Langsamverkehr/pedestrian_all_years_filtered.csv")
    bike_sensors_filtered = filtered_bike_sensors['FK_STANDORT'].unique()
    ped_sensors_filtered = filtered_ped_sensors['FK_STANDORT'].unique()
    
    # Add column for use in model
    active_modes_sensors['model_use'] = False
    active_modes_sensors.loc[active_modes_sensors['sensor_id'].isin(bike_sensors_filtered), 'model_use'] = True
    active_modes_sensors.loc[active_modes_sensors['sensor_id'].isin(ped_sensors_filtered), 'model_use'] = True
    
    # Count None values in the transport_mode column
    none_count = active_modes_sensors['transport_mode'].isna().sum()
    print(f"Number of sensors with unknown transport mode: {none_count}")
    # Drop the sensors with unknown transport mode
    active_modes_sensors = active_modes_sensors.dropna(subset=['transport_mode'])
    
    return active_modes_sensors

def get_plot_sensors(data, model_use=False):
    """
    Load pedestrian sensors data from a CSV file.
    """
    # Plot with color for transport mode
    fig, ax = plt.subplots(figsize=(10, 10))
    # Add color for transport mode
    if model_use:
        # set 'transport mode' to "Insufficient data" if model_use is False
        data.loc[data['model_use'] == False, 'transport_mode'] = "Insufficient data"
    print(model_use)
    print(data['transport_mode'].unique())
    # Get unique transport modes
    modes = data['transport_mode'].unique()
    # define colors for each mode
    colors = defaultdict(lambda: 'black', {
        'bike': 'red',
        'pedestrian': 'darkblue',
        'road': 'dodgerblue',
        'bus': 'green',
        'Insufficient data': 'gray',
    })    
    
    for mode, group in data.groupby('transport_mode'):
        if mode in colors:
            group.plot(ax=ax, color=colors[mode], alpha=0.8, markersize=10, label=mode)
    
    ctx.add_basemap(ax, crs=data.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    plt.title("Sensor locations for all transport modes")
    # remove axis
    ax.set_axis_off()
    # Add legend
    #legend_patches = [Patch(color=color, label=mode) for mode, color in colors.items()]
    ax.legend(loc='upper right', title="Transport mode")       
    if model_use:
        plt.savefig("data/plots/maps/sensors_map_model_use.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig("data/plots/maps/sensors_map.png", dpi=300, bbox_inches='tight')
        
    plt.show()
    # Load the pedestrian sensors data
    
    


def get_all_sensor_locations():
    """
    Load sensor locations from a CSV file.
    """
    # Load the sensor locations data
    road_sensors = get_road_sensors()
    active_mode_sensors = get_active_modes_sensors()
    
    # Combine the two dataframes
    all_sensors = pd.concat([road_sensors, active_mode_sensors], ignore_index=True)
    all_sensors.to_csv("data/processed/positions_sensors_all.csv", index=False)
    
    # Plot the sensors
    get_plot_sensors(all_sensors, model_use=False)
    get_plot_sensors(all_sensors, model_use=True)
    
def add_buffer_to_bounds(bounds, buffer=0.1):
    """
    Add a buffer to the bounds.
    
    Parameters:
    bounds (tuple): A tuple of (minx, miny, maxx, maxy).
    buffer (float): The buffer to add to the bounds.
    
    Returns:
    tuple: A tuple of (minx, miny, maxx, maxy) with the buffer added.
    """
    minx, miny, maxx, maxy = bounds
    buffer_x = buffer * (maxx - minx)
    buffer_y = buffer * (maxy - miny)
    # Add buffer to the bounds
    minx -= buffer_x
    miny -= buffer_y
    maxx += buffer_x
    maxy += buffer_y
    return (minx, miny, maxx, maxy)

def get_network_type(mode):
    if mode == 'bike':
        return 'bike'
    elif mode == 'pedestrian':
        return 'walk'
    elif mode == 'road':
        return 'drive'
    
def create_adj_matrix():
    position = pd.read_csv("data/processed/positions_sensors_all.csv")
    
    # Only keep sensors that are used in the model
    position = position[position['model_use'] == True]
    position["geometry"] = position["geometry"].apply(wkt.loads)
    
    # Create gpd from position
    position = gpd.GeoDataFrame(position, geometry="geometry", crs="EPSG:2056")
    # Change to wgs84 to work with osm
    position = position.to_crs("EPSG:4326")

    # Get unique transport modes
    modes = position['transport_mode'].unique()
    for mode in modes:
        # Create adjacency matrix for each mode
        mode_position = position[position['transport_mode'] == mode]
        
        # Check len of mode_position
        if len(mode_position) < 2:
            print(f"Not enough sensors for mode {mode} to create adjacency matrix. Skipping...")
            A = np.ones((2, 2))  # Create a dummy adjacency matrix
            D = np.ones((2, 2))
            np.save(f"data/train_test/adjacency_matrix_{mode}.npy", A)
            np.save(f"data/train_test/degree_matrix_{mode}.npy", D)
            continue
        
        # Get coordinates border
        bounds = mode_position.total_bounds # returns (left, bottom, right, top)
        # Print the bounds
        bounds = add_buffer_to_bounds(bounds, buffer=0.1)

        network_type = get_network_type(mode)
        
        # Download street network from OSM
        G = ox.graph_from_bbox(bounds, network_type=network_type, simplify=True) #(left, bottom, right, top) network_type (str) – {“all”, “all_public”, “bike”, “drive”, “drive_service”, “walk”}

        # Iterate over the points and find the nearest nodes in the graph and add as new column
        for index, row in mode_position.iterrows():
            point = row['geometry']
            node = ox.distance.nearest_nodes(G, point.x, point.y)
            # Check if node and if node is closer than 100m
            if node is not None:
                # Check if the node is within 10m of the point
                node_point = Point(G.nodes[node]['x'], G.nodes[node]['y'])
                if point.distance(node_point) <= 10:
                    mode_position.at[index, 'nearest_node'] = node
                else:
                    mode_position.at[index, 'nearest_node'] = None
                    print(f"Node {node} is too far from point {point} (distance: {point.distance(node_point)} m)")
            else:
                mode_position.at[index, 'nearest_node'] = None
                print(f"No nearest node found for point {point}")
        
        sensor_nodes = mode_position['nearest_node'].to_list()
        
        # 1. Compute Voronoi cells (NNVD)
        cells = nx.voronoi_cells(G, sensor_nodes, weight='length')

        # 2. Find adjacent regions
        adjacent_pairs = set()
        for u, v in G.edges():
            # Find which sensor node is closest to u and v
            closest_u = None
            closest_v = None
            for center, cell in cells.items():
                if u in cell:
                    closest_u = center
                if v in cell:
                    closest_v = center
            if closest_u != closest_v and closest_u is not None and closest_v is not None and closest_u != "unreachable" and closest_v != "unreachable":
                adjacent_pairs.add(frozenset({closest_u, closest_v}))
        
        
        # Get the sensor list from all sensors
        if mode == 'road':
            initial_index = pd.read_csv("data/processed/MIV/traffic_all_years_filtered.csv")
            initial_index = initial_index["ID"].to_list()
            # For all elements in the list only keep the first 4 characters
            initial_index = [str(x)[:4] for x in initial_index]
            print(f"Number of sensors for mode {mode}: {len(initial_index)} - {initial_index[:5]} ...")
        elif mode == 'bike':
            initial_index = pd.read_csv("data/processed/Langsamverkehr/bike_all_years_filtered.csv")
            initial_index = initial_index["FK_STANDORT"].to_list()
            # all elements to strings
            initial_index = [str(x) for x in initial_index]
            print(f"Number of sensors for mode {mode}: {len(initial_index)} - {initial_index[:5]} ...")
        elif mode == 'pedestrian':
            initial_index = pd.read_csv("data/processed/Langsamverkehr/pedestrian_all_years_filtered.csv")
            initial_index = initial_index["FK_STANDORT"].to_list()
            print(f"Number of sensors for mode {mode}: {len(initial_index)} - {initial_index[:5]} ...")
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Map sensor_postition["nearest_node"] to sensor_list by sensor_position['sensor_id']
        # Create mapping dict from DataFrame
        id_to_node = dict(zip(mode_position['sensor_id'], mode_position['nearest_node']))

        # Map to new list of graph node indexes
        mapped_nodes = [id_to_node[i] for i in initial_index if i in id_to_node]
        print(f"Number of sensors in graph for mode {mode}: {len(mapped_nodes)} - {mapped_nodes[:5]} ...")
        
        # 3. Build adjacency matrix
        sensor_index = defaultdict(list)
        for idx, node in enumerate(mapped_nodes):
            sensor_index[node].append(idx)

        A = np.zeros((len(mapped_nodes), len(mapped_nodes)), dtype=int)
        for pair in adjacent_pairs:
            u, v = pair
            
            for i in sensor_index[u]:
                for j in sensor_index[v]:
                    A[i, j] = 1
                    A[j, i] = 1
            
            if len(sensor_index[u]) > 1:
                for i in sensor_index[u]:
                    for j in sensor_index[u]:
                        if i != j:
                            A[i, j] = 1
                            A[j, i] = 1
            if len(sensor_index[v]) > 1:
                for i in sensor_index[v]:
                    for j in sensor_index[v]:
                        if i != j:
                            A[i, j] = 1
                            A[j, i] = 1
        
        print(f"Number of edges in adjacency matrix for mode {mode}: {np.count_nonzero(A)} (out of {A.size**2})")
        
        
        weight_matrix = np.zeros((len(mapped_nodes), len(mapped_nodes)), dtype=float)
        # 4. Create weight matrix based on distance
        # Iterate over mapped nodes with index and entity
        # For loop with tqdm to show progress
        for index_orig, node_orig in tqdm(enumerate(mapped_nodes), total=len(mapped_nodes)):
        #for index_orig, node_orig in enumerate(mapped_nodes):
            for idx_dest, node_dest in enumerate(mapped_nodes):
                #print(f"Calculating distance between {node_orig} (index {index_orig}) and {node_dest} (index {idx_dest})")
                distance = nx.shortest_path_length(G, node_orig, node_dest, weight='length')
                #print(f"Distance between {node_orig} and {node_dest}: {distance}")
                weight_matrix[index_orig, idx_dest] = np.exp(-distance / 10) if distance > 0.5 else 0.0
        
        np.save(f"data/train_test/weight_matrix_{mode}.npy", weight_matrix)
        print(f"Shape of weight matrix for mode {mode}: {weight_matrix.shape}")
        
        
        # Create a degree matrix D from the adjacency matrix A
        D = np.diag(np.sum(A, axis=1))
        
        # store both matrices as numpy arrays
        np.save(f"data/train_test/adjacency_matrix_{mode}.npy", A)
        np.save(f"data/train_test/degree_matrix_{mode}.npy", D)
        
        print(f"Shape of adjacency matrix for mode {mode}: {A.shape}")
        
        """
        
        # Plot the network
        # Based on the sensor coordinates from gpd mode_position and the A matrix, plot the network
        positions = {i: (mode_position.iloc[i]['geometry'].x, mode_position.iloc[i]['geometry'].y) for i in range(len(sensor_nodes))}

        # 3. Create graph and add edges according to A
        G_plot = nx.Graph()
        G_plot.add_nodes_from(range(len(sensor_nodes)))  # use indices for simplicity
        for i in range(len(sensor_nodes)):
            for j in range(i+1, len(sensor_nodes)):
                if A[i,j] == 1:  # or >0 if weighted
                    G_plot.add_edge(i, j)

        # 4. Plot the graph with coordinates
        plt.figure(figsize=(10, 8))
        nx.draw(G_plot, pos=positions, node_size=50, node_color='red', edge_color='gray', with_labels=True)
        plt.show()
        """
