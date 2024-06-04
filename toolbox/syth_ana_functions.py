# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm
# import matplotlib.colors as colors

# def causal_matshow_ax(causal_results, ax, X, Y, title='Causal Results',ylabel='Causal Strength'):
#     cmap = ListedColormap(['white', 'green'])  # white for low confidence, green for high confidence
#     bounds = [-0.5, 0.5, 1.5]
#     norm = colors.BoundaryNorm(bounds, cmap.N)
    
#     cax = ax.matshow(causal_results, interpolation='nearest', cmap=cmap, norm=norm)
#     plt.colorbar(cax, ax=ax, ticks=[0, 1], shrink=0.6)
#     # lags = np.arange(causal_results.shape[1])  # Assuming lags increase by 1
#     # Y = np.linspace(np.min(Y), np.max(Y), causal_results.shape[0])  # Assuming linear space for causal strengths
#     # add grid lines
#     ax.grid(linestyle='--', linewidth=0.5)
    
#     ax.set_xticks(X)
#     ax.set_yticks(np.linspace(0, len(Y)-1, len(Y)))
#     ax.set_yticklabels([f'{y:.0e}' for y in Y])  # Format labels in scientific notation
    
#     ax.set_xlabel('Lags (yr)')
#     ax.set_ylabel(ylabel)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_title(title)
#     # set the xticklabel to be xticks*10
#     ax.set_xticklabels([str(int(i*10)) for i in ax.get_xticks()])
#     # set the linewidth of the box spine
#     for axis in ['top','bottom','left','right']:
#         ax.spines[axis].set_linewidth(1.5)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as colors

# Testing the updated function with the grid lines aligned to the cell boundaries
def causal_matshow_ax(causal_results, counts, ax, X, Y, title='Causal Results', ylabel='Causal Strength', show_counts=False):
    cmap = ListedColormap(['white', 'green'])  # white for low confidence, green for high confidence
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    cax = ax.matshow(causal_results, interpolation='nearest', cmap=cmap, norm=norm)
    plt.colorbar(cax, ax=ax, ticks=[0, 1], shrink=0.6)
    
    # Set ticks to be at the boundaries of the cells
    ax.set_xticks(np.arange(-0.5, len(X), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(Y), 1), minor=True)
    
    # Use minor ticks to draw grid lines at cell boundaries
    ax.grid(which='minor', color='grey', linestyle='--', linewidth=0.5)

    # Major ticks should be at the center of each cell for the labels
    ax.set_xticks(np.arange(len(X)))
    ax.set_yticks(np.arange(len(Y)))
    ax.set_yticklabels([f'{y:.0e}' for y in Y])  # Format labels in scientific notation
    
    ax.set_xlabel('Lags (yr)')
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(title)
    ax.set_xticklabels([str(int(i*10)) for i in ax.get_xticks()])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    if show_counts:
        # Adding text labels for counts
        for i in range(causal_results.shape[0]):
            for j in range(causal_results.shape[1]):
                # check if it is nan
                if np.isnan(counts[i, j]):
                    ax.text(j, i, ' ', fontsize=8, va='center', ha='center', color='black')
                else:
                    ax.text(j, i, str(int(counts[i, j])), fontsize=8, va='center', ha='center', color='black')








# functions from John Slattery's work

import numpy as np


def linear_ramp(t, t0=0.0, dt=1.0, y0=0.0, dy=1.0, GS_slope = 0.0, GIS_slope=0.0):
    """Linear Ramp Function

    This function describes the piece-wise linear ramp.

    Parameter
    ---------
    t : np.ndarray
        Time variable
    t0 : float
        Start time of the ramp
    dt : float
        Transition length
    y0 : float
        Function value before the transition
    dy : float
        Hight of the transition
    GS_slope : float
        slope before transition - can be positive or negative
    GIS_slope : float
        slope after transition - can be positive or negative

    Return
    ------
    y : np.ndarray
        Function values of the linear transiton
    """
    lt_t0 = t < t0
    gt_t1 = t > t0 + dt
    condlist = [lt_t0,
                ~np.logical_or(lt_t0, gt_t1),
                gt_t1]
    funclist = [lambda t: y0 + GS_slope * (t - t0),
                lambda t: y0 + dy * (t - t0) / dt,
                lambda t: y0 + dy + GIS_slope * (t - t0 - dt)]
    y = np.piecewise(t, condlist, funclist)
    return y


def sample_ar1(n, alpha, sigma=1.0, x0=0):
    """Generate AR(1) noise for evenely sampled series"""
    x = np.zeros(n)
    x[0] = x0 + sigma * np.random.randn()
    sigma_e = np.sqrt(sigma ** 2 * (1 - alpha ** 2))
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + sigma_e * np.random.randn()
    return x





import numpy as np
import pandas as pd

def gen_bi_directional_data(length, delta=10.0, lag=-3, t0=800.0, dt=50.0, dy=1.0, GS_slope=5e-4, GIS_slope=-1e-3, tau=1.0, beta=1e-4, sigma=0.05):
    """
    Generate time series data demonstrating bi-directional causality.
    
    Args:
    length (int): Length of the time series data.
    delta (float): Time step.
    sigma (float): Standard deviation of the noise.
    alpha, beta, gamma, (float): Coefficients describing the interaction between the series.

    Returns:
    pandas.DataFrame: DataFrame containing bi-directionally linked synthetic data.

    """
    # let lag must be nagative integer
    if lag > 0:
        raise ValueError('Lag must be a negative integer.')

    # let beta to smaller than 1e-4
    if beta > 1e-4:
        raise ValueError('Beta must be smaller than 1e-4.')


    time = np.arange(length, step=delta, dtype='float')
    A = np.zeros(len(time))
    B = np.zeros(len(time))

    A[0] = np.random.normal(0, sigma)
    B[0] = np.random.normal(0, sigma)

    alpha = np.exp(-delta / tau)
    
    for t in range(1, len(time)+lag):
        A[t] = alpha * A[t-1] + beta * B[t+lag] + np.random.normal(0, sigma)
        B[t] = alpha * B[t-1] + beta * A[t+lag] + np.random.normal(0, sigma)

    
    trans_A = linear_ramp(time, t0=t0, dt=dt, y0=0.0, dy=dy, GS_slope=GS_slope, GIS_slope=GIS_slope)

    # trans_B = linear_ramp(time, t0=t0-lag*10, dt=dt, y0=0.0, dy=dy, GS_slope=GS_slope, GIS_slope=GIS_slope)
    trans_B = trans_A


    A=trans_A+A
    B=trans_B+B
    
    df = pd.DataFrame({
        'time': time,
        'A': A,
        'B': B
    })
    
    # drop nan values
    df = df.dropna()

    return df





import numpy as np
import pandas as pd


def gen_single_directional_data(length, delta=10.0, lag=-3, t0=800.0, dt=50.0, dy=1.0, GS_slope=5e-4, GIS_slope=-1e-3, tau=1.0, beta=1e-4, sigma=0.05):
    """
    Generate time series data demonstrating bi-directional causality.
    
    Args:
    length (int): Length of the time series data.
    delta (float): Time step.
    sigma (float): Standard deviation of the noise.
    alpha, beta, gamma, (float): Coefficients describing the interaction between the series.

    Returns:
    pandas.DataFrame: DataFrame containing bi-directionally linked synthetic data.

    """
    # let lag must be nagative integer
    if lag > 0:
        raise ValueError('Lag must be a negative integer.')

    # let beta to smaller than 1e-4
    if beta > 1e-4:
        raise ValueError('Beta must be smaller than 1e-4.')


    time = np.arange(length, step=delta, dtype='float')
    A = np.zeros(len(time))
    B = np.zeros(len(time))

    A[0] = np.random.normal(0, sigma)
    B[0] = np.random.normal(0, sigma)

    alpha = np.exp(-delta / tau)
    
    for t in range(1, len(time)+lag):
        A[t] = alpha * A[t-1] + np.random.normal(0, sigma)
        B[t] = alpha * B[t-1] + beta * A[t+lag] + np.random.normal(0, sigma)

    
    trans_A = linear_ramp(time, t0=t0, dt=dt, y0=0.0, dy=dy, GS_slope=GS_slope, GIS_slope=GIS_slope)

    # trans_B = linear_ramp(time, t0=t0-lag*10, dt=dt, y0=0.0, dy=dy, GS_slope=GS_slope, GIS_slope=GIS_slope)
    trans_B = trans_A


    A=trans_A+A
    B=trans_B+B
    
    df = pd.DataFrame({
        'time': time,
        'A': A,
        'B': B
    })
    
    # drop nan values
    df = df.dropna()

    return df



































import numpy as np
import pandas as pd


def gen_dummy_DO(length, delta=10.0, t0=800.0, dt=50.0, dy=1.0, GS_slope=5e-4, GIS_slope=-1e-3, tau=1.0, sigma=0.05):
    """
    Generate dummy time series data for demonstration purposes.

    Args:
    length (int): Length of the time series data.
    delta (float): Time step.
    t0 (float): Starting time.
    dt (float): Duration of the transition.
    dy (float): Amplitude of the transition.
    GS_slope (float): Slope of the Gaussian-shaped transition.
    GIS_slope (float): Slope of the Glacial-interglacial-shaped transition.
    causal_strength (float): Strength of the causal link.
    sigma (float): Standard deviation of the noise.
    tau (float): Time constant for the autoregressive process.

    Returns:
    pandas.DataFrame: DataFrame containing dummy time series data.
    """
    alpha = np.exp(-delta / tau)
    # time = np.arange(t0, t0 + length * delta, step=delta, dtype='float')
    time = np.arange(length, step=delta, dtype='float')
    trans = linear_ramp(time, t0=t0, dt=dt, y0=0.0, dy=dy, GS_slope=GS_slope, GIS_slope=GIS_slope)
    noise = sample_ar1(len(time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise

    df = pd.DataFrame({
        'time': time,
        'A': synt_trans,
        'trans': trans
    })

    return df

def gen_linked_data(df, lag=10, causal_strength=0.5, delta=10.0, tau=1.0, sigma=0.05):
    """
    Generate causally linked synthetic data.

    Args:
    df (pandas.DataFrame): DataFrame containing synthetic time series data.
    lag (int): Lag for the causal effect. Positive lag means left shifting the series.
    causal_strength (float): Strength of the causal link.
    delta (float): Time step.
    tau (float): Time constant for the autoregressive process.
    sigma (float): Standard deviation of the noise.


    Returns:
    pandas.DataFrame: DataFrame containing causally linked synthetic data.
    """
    synt_trans = df['trans'].values
    alpha = np.exp(-delta / tau)
    # Shift the series
    if lag > 0:
        left_shifted_synt_trans = np.full_like(synt_trans, np.nan)
        left_shifted_synt_trans[:-lag] = synt_trans[lag:]
    if lag < 0:
        left_shifted_synt_trans = np.full_like(synt_trans, np.nan)
        left_shifted_synt_trans[-lag:] = synt_trans[:lag]
    if lag == 0:
        left_shifted_synt_trans=synt_trans

    # Generate AR(1) noise as the base for the second causally linked dataset
    new_noise = sample_ar1(len(synt_trans), alpha=alpha, sigma=sigma, x0=0)

    # Combine the lagged original series with the new AR(1) series to produce the causally linked series
    causally_linked_synt_trans = new_noise + causal_strength * left_shifted_synt_trans

    df['B'] = causally_linked_synt_trans
    return df.dropna()



import numpy as np

def get_link_direction(p_matrix, alpha_level):
    # Initialize the result array with False
    results = np.zeros(2, dtype=int)
    
    # Check for A->B
    # Use the second subarray of the first major subarray and ignore the first element
    if np.any(p_matrix[0, 1, 1:] < alpha_level):
        results[0] = 1
    
    # Check for B->A
    # Use the second subarray of the second major subarray and ignore the first element
    if np.any(p_matrix[1, 0, 1:] < alpha_level):
        results[1] = 1
    
    return results
