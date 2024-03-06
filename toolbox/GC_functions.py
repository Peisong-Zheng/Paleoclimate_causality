import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests,kpss
import matplotlib.pyplot as plt

def is_stationary(series, test_method='ADF'):
    """
    Check if a time series is stationary.

    Parameters:
    - series: NumPy ndarray, the time series data to be tested for stationarity.
    - test_method: string, the method used to test for stationarity. Supports 'ADF' for Augmented Dickey-Fuller test
                   and 'KPSS' for Kwiatkowski-Phillips-Schmidt-Shin test.

    Returns:
    - is_stationary: boolean, True if the series is stationary, False otherwise.
    """
    is_stationary = False

    if test_method == 'ADF':
        # Perform Augmented Dickey-Fuller test
        result = adfuller(series)
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        is_stationary = result[1] <= 0.05

    elif test_method == 'KPSS':
        # Perform KPSS test
        statistic, p_value, _, critical_values = kpss(series, 'c')
        print(f'KPSS Statistic: {statistic}')
        print(f'p-value: {p_value}')
        print('Critical Values:')
        for key, value in critical_values.items():
            print(f'\t{key}: {value}')
        # In KPSS test, the null hypothesis is that the series is stationary
        is_stationary = p_value > 0.05

    # Print result
    if is_stationary:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

    return is_stationary



def make_stationary(df, column_names, test_method='ADF',plot=False):
    """
    Process the specified columns of a DataFrame to ensure stationarity.
    If a series is not stationary, it is differenced up to a maximum of 5 times.

    Parameters:
    - df: pandas DataFrame, the DataFrame containing the time series data.
    - column_names: list of strings, the names of the columns to be processed for stationarity.
    - test_method: string, the method used to test for stationarity. Supports 'ADF' and 'KPSS'.

    Returns:
    - The DataFrame with the processed columns where necessary.
    """
    modified_df = df.copy()  # Copy the DataFrame to avoid modifying the original

    for column in column_names:
        for i in range(1, 6):  # Maximum of 5 differencing operations
            # Ensure the series does not have NaN values
            series = modified_df[column].dropna()
            if is_stationary(series.values, test_method):
                print(f"Series in column '{column}' is stationary after {i-1} differencing operation(s).")
                break
            else:
                if i < 5:  # Only difference if we haven't reached the max differencing order
                    modified_df[column] = modified_df[column].diff().dropna()  # Drop NaN after differencing
                    if plot:
                        plt.figure(figsize=(10, 6))
                        plt.plot(df[column], label='Original')
                        plt.plot(modified_df[column], label='Differenced', linestyle='--')
                        plt.legend()
                        plt.title(f"Original vs Differenced Data for {column} (Order {i})")
                        plt.show()
                else:
                    print(f"Series in column '{column}' is not stationary after 5 differencing operations.")
    # drop NaN values
    modified_df = modified_df.dropna()
    return modified_df









# def check_and_difference(data, column_names, plot=False):
#     """
#     Check if the data in the specified columns are stationary.
#     If not, difference the data to make it stationary.
#     Optionally plot the original and differenced data.

#     Parameters:
#     - data: pandas DataFrame containing the time series data.
#     - column_names: list of strings, the names of the columns to check and difference.
#     - plot: bool, whether to plot the original and differenced series for each column.

#     Returns:
#     - modified_data: pandas DataFrame with the differenced series where necessary.
#     """
#     # Copy the input DataFrame to avoid modifying the original data
#     modified_data = data.copy()

#     for column_name in column_names:
#         # Perform Augmented Dickey-Fuller test
#         result = adfuller(modified_data[column_name])
#         print(f'ADF Statistic for {column_name}: {result[0]}')
#         print(f'p-value for {column_name}: {result[1]}')
#         print('Critical Values:')
#         for key, value in result[4].items():
#             print(f'\t{key}: {value}')

#         # Check if stationary
#         if result[1] > 0.05:
#             print(f"Data in {column_name} is not stationary. Differencing the data...")
#             modified_data[column_name] = modified_data[column_name].diff().dropna()
#         else:
#             print(f"Data in {column_name} is stationary.")

#         # Plot if required
#         if plot:
#             plt.figure(figsize=(10, 6))
#             plt.plot(data[column_name], label='Original')
#             plt.plot(modified_data[column_name], label='Differenced', linestyle='--')
#             plt.legend()
#             plt.title(f"Original vs Differenced Data for {column_name}")
#             plt.show()

#     return modified_data.dropna()



def gc_test(data, column_x, column_y, max_lags=4):
    """
    Perform Granger Causality test in both directions and format the result.
    
    Parameters:
    - data: pandas DataFrame containing the time series data.
    - column_x, column_y: names of the columns to be tested.
    - max_lags: int, maximum number of lags to test for.
    
    Returns:
    - Formatted result as a string.
    """
    result_xy = grangercausalitytests(data[[column_x, column_y]], max_lags, verbose=False)
    result_yx = grangercausalitytests(data[[column_y, column_x]], max_lags, verbose=False)
    
    # Initialize variables to store the best p-value and corresponding F statistic and lag
    best_p_xy, best_f_xy, best_lag_xy = 1, None, None
    best_p_yx, best_f_yx, best_lag_yx = 1, None, None
    
    for lag in range(1, max_lags + 1):
        # XY direction
        f_test_xy = result_xy[lag][0]['ssr_ftest']
        if f_test_xy[1] < best_p_xy:
            best_p_xy, best_f_xy, best_lag_xy = f_test_xy[1], f_test_xy[0], lag
        # YX direction
        f_test_yx = result_yx[lag][0]['ssr_ftest']
        if f_test_yx[1] < best_p_yx:
            best_p_yx, best_f_yx, best_lag_yx = f_test_yx[1], f_test_yx[0], lag
    
    # Determine if tests are passed
    test_passed_xy = "True" if best_p_xy < 0.05 else "False"
    test_passed_yx = "True" if best_p_yx < 0.05 else "False"
    
    # Format the result
    result_str = "Granger Causality Test\n------------------------------------------------------------\n"
    result_str += "Direction                   F-statistics         p-value         lag         Granger cause\n"
    result_str += f"{column_x} => {column_y}    {best_f_xy:.3f}               {best_p_xy:.3f}                {best_lag_xy}                {test_passed_xy}\n"
    result_str += f"{column_y} => {column_x}    {best_f_yx:.3f}               {best_p_yx:.3f}                {best_lag_yx}                {test_passed_yx}\n"
    print(result_str)
    
    return result_str



from statsmodels.tsa.arima.model import ARIMA

def find_optimal_ar_order(df, show_figures=False):
    # Determine the maximum order for AR models based on the length of the data
    max_order = min(len(df) // 10, 30)  # Not to exceed 30 to avoid overfitting and computational burden
    
    results = {}  # To store the optimal order and AIC for each column
    
    for column in df.columns:
        best_aic = np.inf
        best_order = None
        for p in range(1, max_order + 1):
            try:
                model = ARIMA(df[column], order=(p, 0, 0))
                model_fit = model.fit()
                aic = model_fit.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = p
            except Exception as e:
                print(f"Could not fit model for {column} with order={p}: {e}")
                break
        
        # Store the best order and AIC for the column
        results[column] = {'Optimal Order': best_order, 'AIC': best_aic}
        
        # If requested, show the comparison figure
        if show_figures and best_order is not None:
            model = ARIMA(df[column], order=(best_order, 0, 0))
            model_fit = model.fit()
            predictions = model_fit.predict()
            
            plt.figure(figsize=(10, 5))
            plt.plot(df[column], label=f'Original {column}', color='blue')
            plt.plot(predictions, label=f'Fitted AR({best_order}) Model', color='red', linestyle='--')
            plt.title(f'Comparison of Original Series and Fitted AR({best_order}) Model for {column}')
            plt.xlabel('Time')
            plt.ylabel(column)
            plt.legend()
            plt.show()
    
    return results




def gc4vars(df, max_lags=10):

    
    gc_results = []  # List to store the results
    columns = df.columns
    
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            column_x = columns[i]
            column_y = columns[j]
            print(f"Testing causality between: {column_x} and {column_y}")
            
            # Call the gc_test function with specified max_lags
            result_str = gc_test(df, column_x, column_y, max_lags=max_lags)
            
            # Parse the result_str to extract causality results
            lines = result_str.split('\n')
            cause_xy = lines[-3].split()[-1]  # Extracts the "True" or "False" for X => Y
            cause_yx = lines[-2].split()[-1]  # Extracts the "True" or "False" for Y => X
            
            # Store the results
            gc_results.append({
                'variables': (column_x, column_y),
                'XY': cause_xy == 'True',
                'YX': cause_yx == 'True'
            })
    
    return gc_results



import networkx as nx
import matplotlib.pyplot as plt

def plot_causal_graph(gc_results, variables=None,dpi=600):
    # Create a directed graph
    G = nx.DiGraph()

    # Infer variables from gc_results if not provided
    if variables is None:
        variables = set()
        for result in gc_results:
            variables.update(result['variables'])
        variables = list(variables)

    # Add nodes (unique variables)
    for var in variables:
        G.add_node(var)

    # Storage for edges to draw them with specific styles later
    edges_uni = []
    edges_bi = []

    # Iterate over the stored results to categorize edges
    for result in gc_results:
        column_x, column_y = result['variables']
        if result['XY'] and result['YX']:  # If causality is bidirectional
            edges_bi.append((column_x, column_y))
            edges_bi.append((column_y, column_x))
        else:
            if result['XY']:  # If X causes Y
                edges_uni.append((column_x, column_y))
            if result['YX']:  # If Y causes X
                edges_uni.append((column_y, column_x))

    for column_x, column_y in edges_uni:
        # For unidirectional edges, add them to the graph with a default color (black) and strength
        G.add_edge(column_x, column_y, strength=1, color='black')

    for column_x, column_y in edges_bi:
        # For bidirectional edges, add them with a distinct color (red)
        G.add_edge(column_x, column_y, strength=1, color='red')

    # Choose layout
    pos = nx.circular_layout(G)

    plt.figure(figsize=(7, 7), dpi=dpi)  # Set the size of the figure

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, edgecolors='black', node_color='lightskyblue', linewidths=1, alpha=1)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edgelist=edges_uni, edge_color='black', arrows=True, width=3, arrowsize=30,
                           min_source_margin=40, min_target_margin=40)
    nx.draw_networkx_edges(G, pos, edgelist=edges_bi, edge_color='red', arrows=True, width=3, arrowsize=30,
                           min_source_margin=40, min_target_margin=40)

    # Labels for the nodes
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.axis('off')  # Turn off the axis
    plt.show()  # Display the graph

