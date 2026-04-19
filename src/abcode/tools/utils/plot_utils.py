import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_corrmat_corrlist(
        df,
        method='spearman',
        sort_corrlist=True,
        csv_fname=None,
        savefig=None,
        plot_corrmat=True,
        plot_clustermap=False,
        use_abs_vals=True,
        annotate_corrmap=False,
        labeltop=False,
        cmap='viridis',
        set_diagonal_to_nan=False
):

    # calculate correlation matrix
    corr_mat = df.corr(numeric_only=True, method=method)
    cols = corr_mat.columns.tolist()
    rows = list(corr_mat.index)
    n = len(cols)

    # # get cols with any NaNs
    # vars_w_nan = []
    # for col in cols:
    #     if corr_mat.loc[:, col].isnull().all():
    #         vars_w_nan.append(col)
    # corr_mat = corr_mat.dropna(how='all', axis=1)
    # corr_mat = corr_mat.dropna(how='all', axis=0)
    # for col in cols:
    #     if col in corr_mat and corr_mat.loc[:, col].isnull().any():
    #         vars_w_nan.append(col)
    #
    # # remove rows and columns with any NaNs
    # cols_nonan = [c for c in cols if c not in vars_w_nan]
    # rows_nonan = [r for r in rows if r not in vars_w_nan]
    # corr_mat = corr_mat.loc[rows_nonan, cols_nonan]
    # cols = cols_nonan
    # rows = rows_nonan

    # get absolute values
    if use_abs_vals:
        corr_mat = corr_mat.abs()

    # save correlation matrix as csv
    if csv_fname is not None:
        corr_mat.round(3).to_csv(f'{csv_fname}.csv')

    # convert correlation matrix into list of pairs and corresponding correlations
    corr_all = {}
    # iterate through every element of corr_mat
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            if row != col:
                pair = [row, col]
                pair.sort()
                pair = tuple(pair)
                if pair not in corr_all:
                    corr_all.update({pair: float(corr_mat.loc[row, col])})

    # sort correlation data in descending order of correlations
    index = []
    corrs = []
    for k, v in corr_all.items():
        index.append(k)
        corrs.append(v)
    corr_all = pd.DataFrame(corrs, columns=['corr'], index=index)
    corr_all['corr_abs'] = corr_all['corr'].abs()
    if sort_corrlist:
        corr_all = corr_all.sort_values(by='corr_abs', ascending=False)

    # set diagonal to zero
    corr_mat_to_plot = corr_mat.copy()
    if set_diagonal_to_nan:
        np.fill_diagonal(corr_mat_to_plot.values, np.nan)

    # plot correlation matrix
    if plot_corrmat:
        fig, ax = plt.subplots(1, 1, figsize=(n/20*8, n/20*8))
        if use_abs_vals:
            heatmap(corr_mat_to_plot.to_numpy(), ax=ax, datamin=0, datamax=1,
                    logscale_cmap=False, annotate=None, row_labels=rows, col_labels=cols, labeltop=labeltop, c=cmap)
        else:
            heatmap(corr_mat_to_plot.to_numpy(), ax=ax, datamin=-1, datamax=1,
                    logscale_cmap=False, annotate=None, row_labels=rows, col_labels=cols, labeltop=labeltop, c=cmap)
        if annotate_corrmap:
            for i in range(corr_mat_to_plot.shape[0]):
                for j in range(corr_mat_to_plot.shape[1]):
                    txt = str(round(corr_mat_to_plot[i,j],2))
                    ax.text(j-0.25,i+0.1, txt, fontsize=7)
        ax.grid(None)
        if savefig is not None:
            fig.savefig(f'{savefig}.png', bbox_inches='tight')

    # get clustermap
    if plot_clustermap:
        cl = sns.clustermap(corr_mat, cmap=cmap, figsize=(12, 12), yticklabels=True, xticklabels=True)
        cl.ax_heatmap.set_yticklabels(cl.ax_heatmap.get_ymajorticklabels(), fontsize=7)
        cl.ax_heatmap.set_xticklabels(cl.ax_heatmap.get_xmajorticklabels(), fontsize=7)
        cl.fig.suptitle('Cluster map of selected features', fontsize=16)
        # plt.title(f'Cluster map of {figtitle}', fontsize=16, loc='center')
        if savefig is not None:
            plt.savefig(f"{savefig}_clustermap.png",  bbox_inches='tight')

    return corr_mat, corr_all


def get_high_correlation_pairs(corr_all, corr_thres, print_out=True):
    # filter by corr_thres
    corr_selected = corr_all[corr_all.corr_abs > corr_thres]
    index_selected = list(corr_selected.index)
    # print list of pairs with high correlations
    if print_out:
        for i in range(len(corr_selected)):
            print(i, f"({index_selected[i][0]}, {index_selected[i][1]}):", round(
                corr_selected.iloc[i]['corr'], 4))
    return corr_selected

def get_dict_of_features_with_highcorr(corr_mat, corr_thres):
    highcorr_vars_dict = {}
    rows = list(corr_mat.index)
    for i, xvar in enumerate(rows):
        corr_mat_row = corr_mat.loc[xvar]
        high_corr_vars = list(
            corr_mat_row[corr_mat_row.abs() > corr_thres].index)
        high_corr_vars.remove(xvar)
        highcorr_vars_dict[xvar] = high_corr_vars
        print(i, xvar, ':', *high_corr_vars)
    return highcorr_vars_dict


def symlog(data):
    idx_pos = np.where(data > 0)
    idx_neg = np.where(data < 0)
    data_symlog = np.zeros((data.shape[0], data.shape[1]))
    data_symlog[:] = np.nan
    data_symlog[idx_pos] = np.log(data[idx_pos])
    data_symlog[idx_neg] = -np.log(-data[idx_neg])
    return data_symlog

def annotate_heatmap(array_2D, ax, ndecimals=2, fontsize=8):
    for (j,i),label in np.ndenumerate(array_2D):
        if ndecimals==0 and ~np.isnan(label):
            label = int(label)
        else:
            label = round(label,ndecimals)
        ax.text(i,j,label,ha='center',va='center', color='0.8', fontsize=fontsize, fontweight='bold')


def heatmap(array, c='virdis', ax=None, cbar_kw={}, cbarlabel="", datamin=None, datamax=None, logscale_cmap=False,
            annotate=None, row_labels=None, col_labels=None, show_gridlines=True, labeltop=False, rotation=90,
            show_colorbar=True, fontsize=8):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()
    cmap = getattr(plt.cm, c)

    # get array size and xy labels
    data = array.astype(float)
    ny, nx = data.shape

    # get row and column labels
    if row_labels is None:
        row_labels = list(np.arange(ny) + 1)
    if col_labels is None:
        col_labels = list(np.arange(nx) + 1)

    # get locations of nan values and negative values, replace values so these don't trigger an error
    naninds = np.where(np.isnan(data) == True)
    infinds = np.where(np.isinf(data) == True)
    if len(infinds[0]) > 0:
        data[infinds] = np.nan
    if len(naninds[0]) > 0:
        data[naninds] = np.nanmean(data)
    if len(infinds[0]) > 0:
        data[infinds] = np.nanmean(data)
    data_cmap = data.copy()

    # get min and max values
    if datamin is None:
        datamin = np.nanmin(data_cmap)
    if datamax is None:
        datamax = np.nanmax(data_cmap)

    # get colormap to plot
    if logscale_cmap:  # plot on logscale
        data_cmap = symlog(data_cmap)
        datamin, datamax = np.min(data_cmap), np.max(data_cmap)

    # get cmap gradations
    dataint = (datamax - datamin) / 100
    norm = plt.Normalize(datamin, datamax + dataint)
    # convert data array into colormap
    colormap = cmap(norm(data_cmap))

    # Set the positions of nan values in colormap to 'lime'
    colormap[naninds[0], naninds[1], :3] = 1, 1, 1  # 0,1,0
    colormap[infinds[0], infinds[1], :3] = 1, 1, 1

    # plot colormap
    im = ax.imshow(colormap, interpolation='nearest')

    # Create colorbar
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.07)
        cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        if logscale_cmap == True:
            cbar_labels = cbar.ax.get_yticks()
            cbar.set_ticks(cbar_labels)
            cbar_labels_unlog = list(np.round(np.exp(np.array(cbar_labels)), 2))
            cbar.set_ticklabels(cbar_labels_unlog)
    else:
        cbar = None

    # Turn off gridlines if required
    ax.tick_params(axis='both', which='both', length=0, gridOn=show_gridlines)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.

    ax.set_yticklabels(row_labels, fontsize=fontsize)

    # Let the horizontal axes labeling appear on top or bottom.
    if labeltop:
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.set_xticklabels(col_labels, fontsize=fontsize, ha="left")
    else:
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        ax.set_xticklabels(col_labels, fontsize=fontsize, ha="right")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=rotation,
             rotation_mode="anchor")

    # Annotate
    if annotate is not None:
        if isinstance(annotate, int):
            ndecimals = annotate
        else:
            ndecimals = 3
        annotate_heatmap(array, ax, ndecimals=ndecimals)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # set xticks
    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar, ax


def plot_variant_heatmap(arr, seq, N_res_per_heatmap_row, aa_list, seq_name=None, savefig=None, figtitle=None, c='bwr'):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    # convert arr to numpy
    num_pos = len(seq)
    pos_list = list(np.arange(1, len(seq) + 1))
    if not isinstance(arr, np.ndarray):
        pos_list = arr.columns.tolist()
        num_pos = len(pos_list)
        arr = arr.to_numpy()

    # obtain heatmap parameters
    num_heatmaps = int(np.ceil(num_pos / N_res_per_heatmap_row))
    heatmap_min = np.min(arr)
    heatmap_max = np.max(arr)
    # define norm for colormap
    if c == 'bwr':
        print('heatmap_min:',heatmap_min, 'heatmap_max:',heatmap_max)
        if heatmap_min < 0 and heatmap_max > 0:
            norm = colors.TwoSlopeNorm(vmin=heatmap_min, vcenter=0, vmax=heatmap_max)
        elif heatmap_min >= 0 and heatmap_max > 0:
            c = 'Reds'
            norm = colors.Normalize(vmin=0, vmax=heatmap_max)
        elif heatmap_max <= 0 and heatmap_min < 0:
            c = 'Blues_r'
            norm = colors.Normalize(vmin=heatmap_min, vmax=0)
    else:
        norm = None
    # define color for NaN elements
    cmap = getattr(plt.cm, c)
    cmap.set_bad('lime')
    # plot heatmap
    fig, ax = plt.subplots(num_heatmaps, 1, figsize=(N_res_per_heatmap_row / len(aa_list) * 4, num_heatmaps * 4))
    for k in range(num_heatmaps):
        if num_heatmaps == 1:
            ax_k = ax
        else:
            ax_k = ax[k]
        pos_list_k = pos_list[k * N_res_per_heatmap_row:min((k + 1) * N_res_per_heatmap_row, num_pos)]
        seq_k = [seq[pos-1] for pos in pos_list_k]
        start_idx = k * N_res_per_heatmap_row
        end_idx = min((k + 1) * N_res_per_heatmap_row, num_pos)
        heatmap_k = arr[:, start_idx:end_idx]
        # Some WT sequences include non-canonical residues (e.g., 'X'). Skip WT marker for those positions.
        wt_pairs = [
            [aa_list.index(wt_aa), res_idx]
            for res_idx, wt_aa in enumerate(seq_k)
            if wt_aa in aa_list
        ]
        wt_idxs_k = np.array(wt_pairs, dtype=int) if wt_pairs else np.empty((0, 2), dtype=int)
        im = ax_k.imshow(heatmap_k, norm=norm, cmap=cmap, aspect="auto")
        # annotate WT amino acid with red dot
        if wt_idxs_k.size > 0:
            ax_k.scatter(wt_idxs_k[:,1], wt_idxs_k[:,0], c='r', s=4)
        ax_k.set_yticks(range(len(aa_list)), aa_list)
        ax_k.set_xticks(range(len(pos_list_k)), pos_list_k, fontsize=7, rotation=45)

    fig.colorbar(im, orientation='vertical')
    if figtitle is not None:
        if seq_name is not None:
            figtitle = seq_name + ': ' + figtitle
        plt.suptitle(figtitle, y=0.93, fontsize=16)
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
