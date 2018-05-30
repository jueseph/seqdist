import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#=================================
# Utility functions
#=================================

def harmonize_indexes(dat1, dat2):
    """Subset and reorder indexes on two datasets so that elements of the indexes map one-to-one.
    """
    idx = np.intersect1d(dat1.index, dat2.index)
    
    dat1 = dat1.loc[idx]
    if len(dat1.shape)>1: # this is a matrix
        dat1 = dat1[idx]

    dat2 = dat2.loc[idx]
    if len(dat2.shape)>1: # this is a matrix
        dat2 = dat2[idx]
        
    return dat1, dat2

def load_tcoffee_dist(filename, as_dist=True):
    """Loads distance matrix output from T-coffee"""

    names = []
    with open(filename) as f:
        for line in f:
            if 'SEQ_INDEX' in line:
                names.append(line.split()[2])
            elif 'PW_SEQ_DISTANCES' in line:
                break

        mat = np.zeros((len(names),len(names)))
        for line in f:
            if ('BOT' in line) or ('TOP' in line):
                tokens = line.split()
                mat[int(tokens[1]),int(tokens[2])] = float(tokens[3])

        for i in range(len(mat)):
            mat[i,i] = 100

    if as_dist:
        mat = (100 - mat)/100

    df = pd.DataFrame(mat)
    df.columns = names
    df.index = names

    return df

def load_clustalo_dist(filename):
    """Loads distance matrix output from Clustal Omega."""

    df = pd.read_table(filename,skiprows=1,header=None, delim_whitespace=True)
    mat = df.values[:,1:]
    names = df.iloc[:,0].values

    df = pd.DataFrame(mat)
    df.columns = names
    df.index = names

    return df

# originally from juetools
def box_off(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

# originally from juetools
def deduplicate_legend_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    return (newHandles, newLabels)


#=================================
# Mantel test functions
#
# The functions below calculate a Pearson or Spearman correlation coefficient
# between genotypic and phenotypic distance, and then repeats the calculation
# for many permutations of the phenotype distance matrix.
#=================================

def mantel_test(geno_dist, 
                pheno_val,
                nperm = 1000, 
                random_state = np.random.RandomState(seed=np.random.randint(100))):
    """Runs Mantel test on one pair of genotypic and phenotypic distance matrices.
    """
    
    from scipy.spatial.distance import pdist, squareform
    import scipy.stats as stats

    # make phenotype distance matrix
    geno_dist, pheno_val = harmonize_indexes(geno_dist, pheno_val)
    pheno_dist = squareform(pdist(np.reshape(pheno_val.values,(len(pheno_val),1))))
    g = squareform(geno_dist) # condense to lower diagonal terms only
    p = squareform(pheno_dist)

    # permute phenotype distance matrix
    p_perm = []

    for i in range(nperm):        
        idx_perm = random_state.permutation(range(len(pheno_dist)))
        pp = pheno_dist[:,idx_perm]
        pp = pp[idx_perm,:]
        p_perm.append(squareform(pp))

    perms = {}
    perms['pearson'] = [stats.pearsonr(g,pp)[0] for pp in p_perm]
    perms['spearman'] = [stats.spearmanr(g,pp).correlation for pp in p_perm]
    
    # return row for adding to table of results
    row = pd.DataFrame.from_items([
        ('num-examples',[len(pheno_val)]),
        ('num-perms',[nperm]),
        ('pearson',[stats.pearsonr(g,p)[0]]),
        ('spearman',[stats.spearmanr(g,p).correlation])
    ])
    
    for corrtype in ['pearson','spearman']:
        row['%s-mean' % corrtype] = np.mean(perms[corrtype])
        row['%s-std' % corrtype] = np.std(perms[corrtype])
        row['%s-zscore' % corrtype] = (row['%s' % corrtype] - row['%s-mean' % corrtype])/row['%s-std' % corrtype]

    return row

def run_multiple_mantel_tests(distmat_info, 
                              pheno_val,
                              nperm = 1000, 
                              random_seed = None,
                              prev_results=None):

    df = pd.DataFrame()
    
    for i,row in distmat_info.iterrows():
        if prev_results is not None:
            tmp = prev_results[prev_results['label']==row['label']]
            if len(tmp)>0:
                print('Using previous results for %s' % row['label'])
                df = df.append(tmp)
                continue
        
        if type(row['small-distance-matrix-path']) is str:
            geno_dist = pd.DataFrame.from_csv(row['small-distance-matrix-path'])
        elif type(row['distance-matrix-path']) is str:
            geno_dist = pd.DataFrame.from_csv(row['distance-matrix-path'])
        else:
            df = df.append(row)
            continue
            
        print('Permutation test on %s' % row['label'])
        extra_cols = mantel_test(geno_dist    = geno_dist,
                                 pheno_val    = pheno_val,
                                 nperm        = nperm,
                                 random_state = np.random.RandomState(random_seed))
        newrow = pd.DataFrame(row).T
        extra_cols.index = newrow.index
        newrow = pd.concat([newrow, extra_cols],axis=1)
        df = df.append(newrow)

    df = df.reset_index(drop=True)

    return df

def plot_correlation_bars_raw(df, corr_type, ax=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=(6,3));
    else:
        fig = None
        
    barsize = df['%s' % corr_type]
    y = range(len(barsize))[::-1]

    ax.barh(y,barsize, alpha=0.5, label='Correlation');
    ax.set_yticks(y);
    ax.set_yticklabels(df['label']);

    ax.errorbar(df['%s-mean' % corr_type], y, xerr=2*df['%s-std' % corr_type],
                fmt='ro',capsize=3,markerfacecolor='none', label='Mean +/- 2 s.d., 1000 permutations');
    yl = ax.get_ylim();
    ax.set_ylim(yl);
    ax.plot([0,0],yl,'k--');

    box_off(ax)
    lgd = ax.legend(*deduplicate_legend_labels(ax),loc='lower center',bbox_to_anchor=(0.5,1.01));
    ax.set_xlabel('Correlation');
    ax.set_ylabel('Genotypic distance metric');
    return fig, ax


def plot_correlation_bars_zscore(df, corr_type, ax=None, leglabel=''):
    if ax is None:
        fig,ax = plt.subplots(figsize=(6,3));
    else:
        fig = None
        
    barsize = df['%s-zscore' % corr_type]
    y = range(len(barsize))[::-1]
    ax.barh(y,barsize, alpha=0.5, label=leglabel, color='C1');
    ax.set_yticks(y);
    ax.set_yticklabels(df['label']);

    yl = ax.get_ylim();
    ax.set_ylim(yl);
    ax.plot([0,0],yl,'k--');

    box_off(ax)
#     lgd = ax.legend(*deduplicate_legend_labels(ax),loc='upper left');
    ax.set_xlabel('Z-score');
    ax.set_ylabel('Genotypic distance metric');
    return fig, ax

#=================================
# k-NN functions
#
# The functions below train k-nearest-neighbor models on the distance matrix
# and phenotype values, and plot the R2 of predictions versus measurements.
#=================================

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

def run_knn(distmat_info, k_range, pheno_val):
    
    results = pd.DataFrame()

    for i,row in distmat_info.iterrows():

        if type(row['small-distance-matrix-path']) is str:
            dist = pd.DataFrame.from_csv(row['small-distance-matrix-path'])
        elif type(row['distance-matrix-path']) is str:
            dist = pd.DataFrame.from_csv(row['distance-matrix-path'])
        else:
            results = results.append(row)
            continue

        X, y = harmonize_indexes(dist, pheno_val)

        newrow = row.copy()
        for k in k_range:
            neigh = KNeighborsRegressor(n_neighbors=k, metric='precomputed')

            yp = cross_val_predict(neigh, X.values, y.values, cv=10)

            newrow['R2 k=%d' % k] = r2_score(y,yp)
            newrow['MSE k=%d' % k] = mean_squared_error(y,yp)

        results = results.append(newrow)
    
    return results

def plot_knn_results_3panel(results, axs=None):

    if axs is None:
        fig,axs = plt.subplots(1,3,figsize=(9.5,3));
        plt.subplots_adjust(wspace=0.4);

    for i,row in results.iterrows():

        ax = axs[0]

        tmp = row[row.index.str.contains('R2')]
        x = tmp.index.str.extract('k=(\d*)',expand=False).astype(int)
        y = tmp.values

        i = np.argsort(x)
        x = x[i]
        y1 = y.T[i]

        ax.plot(x,y1, 'o-');
        ax.set_ylabel('R2');
        ax.set_xlabel('k');

        ax = axs[1]

        tmp = row[row.index.str.contains('MSE')]
        x = tmp.index.str.extract('k=(\d*)',expand=False).astype(int)
        y = tmp.values

        i = np.argsort(x)
        x = x[i]
        y2 = y.T[i]

        ax.plot(x,y2, 'o-');
        ax.set_ylabel('MSE');
        ax.set_xlabel('k');
        
        ax = axs[2]
        ax.plot(y1,y2,'o-')
        ax.set_ylabel('MSE');
        ax.set_xlabel('R2');

    axs[0].set_xscale('log');
    axs[1].set_xscale('log');

    return fig, axs

def get_max_R2(results):
    newdf = pd.DataFrame()

    for i,row in results.iterrows():
        tmp = row[row.index.str.contains('R2')]
        x = tmp.index.str.extract('k=(\d*)',expand=False).astype(int)
        y = tmp.values

        newrow = row[~row.index.str.contains('R2') & ~row.index.str.contains('MSE')].copy()
        newrow['best k'] = x[np.argmax(y)]
        newrow['R2'] = max(y)
        newdf = newdf.append(newrow)
    
    return newdf

def plot_R2_bars(results):

    fig,ax = plt.subplots(figsize=(4,6));

    x = results['R2']
    y = range(len(x))[::-1]
    ax.barh(y, x, alpha=0.5, color='C1');
    ax.set_yticks(y);
    ax.set_yticklabels(results['label']);

    box_off(ax)
    ax.set_xlabel('$R^2$');
    ax.set_ylabel('Genotypic distance metric');
    
    return fig, ax

