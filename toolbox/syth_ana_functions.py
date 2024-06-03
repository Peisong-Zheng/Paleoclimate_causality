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
