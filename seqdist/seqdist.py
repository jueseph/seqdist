#!/usr/bin/env python

import numpy as np
import pandas as pd
from Bio import AlignIO

def count_identities(seqs, subsmat):
    # TODO: This is 3x slower than count_similarities below. Fix this.
    return sum((seqs[0] == seqs[1]) & ((seqs[0] != '-') | (seqs[1] != '-')))

def count_similarities(seqs, subsmat):
    sims = [subsmat.get((c1,c2),0) for c1,c2 in zip(seqs[0],seqs[1])]
    return sum(sims)

def length_min(seqs):
    return min(sum(seqs[0] != '-'), sum(seqs[1] != '-'))

def length_max(seqs):
    return max(sum(seqs[0] != '-'), sum(seqs[1] != '-'))

def length_nonorm(seqs):
    return 1

def length_overlap(seqs):
    s1gap = seqs[0] == '-'
    s2gap = seqs[1] == '-'
    idxnongap1 = np.where(~s1gap)[0]
    idxnongap2 = np.where(~s2gap)[0]
    left = max(idxnongap1[0], idxnongap2[0])
    right = min(idxnongap1[-1], idxnongap2[-1])
    numbothgaps = sum(s1gap[left:(right+1)] & s2gap[left:(right+1)])
    return right - left + 1 - numbothgaps

def calc_mat_parallel(aln_array, subsmat, 
                      metric_func=count_identities, 
                      length_func=length_overlap,
                      num_procs=1):

    import multiprocessing
    from functools import partial
    from scipy.spatial.distance import squareform

    pool = multiprocessing.Pool(num_procs)

    metric_func_helper = partial(metric_func, subsmat=subsmat)
    
    # off-diagonal elements of dist. matrix
    N = len(aln_array)
    seqs = [(aln_array[i], aln_array[j]) for i in range(N) for j in range(i+1,N)]
    counts = pool.map(metric_func_helper, seqs)
    lengths = pool.map(length_func, seqs)
    
    mat = [float(c)/L if L>0 else 0 for c,L in zip(counts,lengths)]        
    mat = squareform(mat)
    
    # diagonal elements
    seqs = [(aln_array[i], aln_array[i]) for i in range(N)]
    counts = pool.map(metric_func_helper, seqs)
    lengths = pool.map(length_func, seqs)
    
    for i in range(N):
        mat[i,i] = float(counts[i])/lengths[i]
        
    return mat

def simscore_to_distance(distmat):

    distmat = distmat.copy()

    for i in range(len(distmat)):
        for j in range(i+1,len(distmat)):
            # enforces min value of 0, as some pathological cases lead to negative distances
            distmat[i,j] = max(min(distmat[i,i],distmat[j,j]) - distmat[i,j],0)
            distmat[j,i] = distmat[i,j]
        distmat[i,i] = 0
        
    return distmat

def make_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate percent identity or similarity from a multiple-sequence alignment.\nDefault arguments replicate calculations in Geneious.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('msa', metavar='ALIGNMENT.afa', nargs='?', default=None,
                        help='Multiple sequence alignment (in fasta format).')
    group.add_argument('--list-subsmat', action='store_true', 
                        help='List available substitution matrices (from Bio.SubsMat.MatrixInfo) for computing similarity.')
    parser.add_argument('-o','--distmat-out', metavar='msa.dist.csv', 
                        help='File to save distance matrix (pandas DataFrame csv). Default: stdout.')
    parser.add_argument('-n','--num-procs', default=1, type=int,
                        help='Number of processor cores to use for calculations.')
    parser.add_argument('-m','--mode', choices=['id','nonid','sim','nonsim','simscore','simscore-dist'], default='nonid', 
                        help='Length-normalized (non-)identity or (non-)similarity, where "non-identity" = 1-identity. "simscore" is a sum of substitution matrix scores; "simscore-dist" is the sum converted to a distance. Default: nonid')
    parser.add_argument('--subsmat', default='blosum62', 
                        help='Substitution matrix to determine similarity (when using "-m sim" or "-m nonsim"). Default: blosum62')
    parser.add_argument('--score-thresh', default=0, 
                        help='Amino-acid substitutions with scores > thresh are considered similar (when using "-m sim" or "-m nonsim"). Default: 0')
    parser.add_argument('--length-norm', choices=['min','max','overlap','none'], default='overlap', 
                        help='Length by which to normalize identity or similarity counts. Default: overlap')
    return parser


def main():
    from Bio.SubsMat import MatrixInfo
    import re

    p = make_argparser()
    args = p.parse_args()

    if args.list_subsmat:
        print('\nAvailable substitution matrices:')
        print(', '.join([m for m in MatrixInfo.available_matrices]))
        print('')
        import sys
        sys.exit()
        
    if args.length_norm == 'max':
        length_func = length_max
    elif args.length_norm == 'min':
        length_func = length_min
    elif args.length_norm == 'overlap':
        length_func = length_overlap
    elif args.length_norm == 'none':
        length_func = length_nonorm

    # load alignment
    aln = AlignIO.read(args.msa,'fasta')
    accessions = [s.id for s in aln]

    aln = [str(s.seq) for s in aln]                  # convert to string
    aln = [s.replace('.','-') for s in aln]          # standardize gap character
    aln = [re.sub('[a-z]','-',seq) for seq in aln]   # remove unaligned residues
    aln_array = [np.array(list(seq)) for seq in aln] # convert to numpy array

    if args.mode == 'id' or args.mode == 'nonid':
        subsmat = None
        metric_func = count_identities
        
    elif args.mode == 'sim' or args.mode == 'nonsim':
        subsmat = eval('MatrixInfo.'+args.subsmat).copy()
        subsmat.update({(k[1],k[0]):v for k,v in subsmat.items()}) # make symmetric
        subsmat = {k:(1 if v > args.score_thresh else 0) for k,v in subsmat.items()} # threshold
        metric_func = count_similarities
    
    elif 'simscore' in args.mode:
        subsmat = eval('MatrixInfo.'+args.subsmat).copy()
        subsmat.update({(k[1],k[0]):v for k,v in subsmat.items()}) # make symmetric
        metric_func = count_similarities
        
    dm = calc_mat_parallel(aln_array, subsmat=subsmat, 
                           metric_func=metric_func, 
                           length_func=length_func,
                           num_procs=args.num_procs)

    if 'non' in args.mode:
        dm = 1-dm

    if args.mode == 'simscore-dist':
        dm = simscore_to_distance(dm)

    df = pd.DataFrame(dm)
    df.insert(0,'accession',accessions)
    df.columns = ['accession']+accessions

    if args.distmat_out:
        df.to_csv(args.distmat_out,index=None)
    else:
        from StringIO import StringIO
        output = StringIO()
        df.to_csv(output, index=None)
        print(output.getvalue())
        
