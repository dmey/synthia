import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def plot_random_columns(ds_true, ds_synth, n_columns=10):
    for name in ds_true:
        if ds_true[name].ndim != 2:
            continue
        level = ds_true[name].dims[1]
        fix, ax = plt.subplots(1,2, figsize=(15,5))
        rnd = np.random.choice(ds_true['column'], n_columns)
        ds_true[name][rnd,:].plot.line(x=level, ax=ax[0])
        ax[0].legend('')
        ax[0].title.set_text('True')
        # Only if the two datasets have not the same size
        # we need to generate new random indices.
        if len(ds_true.column) != len(ds_synth.column):
            rnd = np.random.choice(ds_synth['column'], n_columns)
        ds_synth[name][rnd,:].plot.line(x=level, ax=ax[1])
        ax[1].legend('')
        ax[1].title.set_text('Pred')
        plt.show()
        d1_synthetic = np.mean(np.abs(np.diff(ds_synth[name])))
        d1_true = np.mean(np.abs(np.diff(ds_true[name])))
        d2_synthetic = np.mean(np.abs(np.diff(ds_synth[name], n = 2)))
        d2_true = np.mean(np.abs(np.diff(ds_true[name], n = 2)))
        print(f'First derivatives for synthetic: {d1_synthetic}, true: {d1_true}')
        print(f'Second derivatives for synthetic: {d2_synthetic}, true: {d2_true}')

def plot_ds_hist(ds_true, ds_synth):
    for name in ds_true:
        if ds_true[name].ndim != 1:
            continue
        ds_true[name].plot.hist(label='True', alpha=0.5, bins=100)
        ds_synth[name].plot.hist(label='Synthetic', alpha=1, bins=100)
        plt.legend()
        plt.show()


def plot_summary_stat_column(da, da_2=None):
    if da_2 is None:
        cols = 1
    else:
        cols = 2

    fig, ax = plt.subplots(1,cols, figsize=(10*cols,5))

    if da_2 is None:
        ax_1 = ax
    else:
        ax_1 = ax[0]
        ax_2 = ax[1]

    da.min('column').plot(label='min', ax=ax_1)
    da.max('column').plot(label='min', ax=ax_1)
    da.mean('column').plot(label='mean', ax=ax_1)
    da.sel(column=np.random.choice(da.column)).plot(label='random column', ax=ax_1)
    ax_1.legend()
    if da_2 is not None:
        da_2.min('column').plot(label='min', ax=ax_2)
        da_2.max('column').plot(label='min', ax=ax_2)
        da_2.mean('column').plot(label='mean', ax=ax_2)
        da_2.sel(column=np.random.choice(da.column)).plot(label='random column', ax=ax_2)
        ax_2.legend()
