import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as colors

def causal_matshow_ax(causal_results, ax, title='Causal Results',ylabel='Causal Strength'):
    cmap = ListedColormap(['white', 'green'])  # white for low confidence, green for high confidence
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    cax = ax.matshow(causal_results, interpolation='nearest', cmap=cmap, norm=norm)
    plt.colorbar(cax, ax=ax, ticks=[0, 1], shrink=0.6)
    lags = np.arange(causal_results.shape[1])  # Assuming lags increase by 1
    causal_strengths = np.linspace(0.1, 1.0, causal_results.shape[0])  # Assuming linear space for causal strengths
    # add grid lines
    ax.grid(linestyle='--', linewidth=0.5)
    
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels(lags)
    ax.set_yticks(np.arange(len(causal_strengths)))
    ax.set_yticklabels(np.round(causal_strengths, 1))
    ax.set_xlabel('Lags (yr)')
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(title)
    # set the xticklabel to be xticks*10
    ax.set_xticklabels([str(int(i*10)) for i in ax.get_xticks()])
    # set the linewidth of the box spine
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)







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

def gen_bi_directional_data(length, delta=10.0, lag=10, t0=800.0, dt=50.0, dy=1.0, GS_slope=5e-4, GIS_slope=-1e-3, tau=1.0, beta=1e-4, sigma=0.05):
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
