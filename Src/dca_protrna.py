################################################################################
#BSD 2-Clause License                                                          #
#                                                                              #
#Copyright (c) 2019, Vladimir Reinharz                                         #
#All rights reserved.                                                          #
#                                                                              #
#Redistribution and use in source and binary forms, with or without            #
#modification, are permitted provided that the following conditions are met:   #
#                                                                              #
#* Redistributions of source code must retain the above copyright notice, this #
#  list of conditions and the following disclaimer.                            #
#                                                                              #
#* Redistributions in binary form must reproduce the above copyright notice,   #
#  this list of conditions and the following disclaimer in the documentation   #
#  and/or other materials provided with the distribution.                      #
#                                                                              #
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"   #
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE     #
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE#
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE  #
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL    #
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR    #
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER    #
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, #
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE #
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          #
################################################################################

import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from itertools import combinations, product, chain
import os
from multiprocessing import Pool
import pickle
from pprint import pprint
from statistics import mean, stdev
from time import time


import pandas as pd
import numpy as np
from tqdm import tqdm


PSEUDOCOUNT_WEIGHT = 0.5



LEN_PROT = 0
LEN_RNA = 0
GAPS = 0
CONSERVATION = 0
D_FREQ_ONE = {}

SYMBOLS = tuple(('-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
              'a', 'c', 'g', 'u', '.'))
SYMBOLS_PROT = tuple(('-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'))
SYMBOLS_RNA = tuple(('a', 'c', 'g', 'u', '.'))
MAP_SYMBOLS = {x:i for i, x in enumerate(SYMBOLS)}
DF = None
NB_PROCS = 6

MA = None
MEFF = None





map_key = 0 
INV_C = 0


@contextmanager
def timer(name):
    tic = time()
    yield
    toc = time()
    print(f"{name} took {toc - tic:.2f}s")


def read_merged_fasta(path):
    """Reads the 'fasta' file, makes sure only characters
    in admissible alphabets are in, and also check lens of prots and rnas
    """
    d = defaultdict(str)
    len_prot = 0
    len_rna = 0

    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].strip()
            else:
                prot, rna = tuple(line.strip().split())
                prot = prot.upper().replace('.', '-')
                rna = rna.lower()

                if (any(x not in SYMBOLS for x in prot) or
                    any(x not in SYMBOLS for x in rna)):
                    continue

                d[name] = (prot, rna)
    len_prot, len_rna = list(map(len, next(iter(d.values()))))
    return d, len_prot, len_rna


def make_df(data):
    """Makes dataframe of the concatenated alignment"""

    values = []
    index = []
    for k, v in data.items():
        index.append(k)
        prot, rna = v
        values.append(list(prot + rna))
    return pd.DataFrame(values, index=index)



"""
The next four methods compute the frequency of each positions, pairs
and with the 'position correction' (pc).
"""

def freq_one(pos, x):
    if pos < LEN_PROT:
        l_ma = MA[:,0]
        meff = MEFF[0]
    else:
        l_ma = MA[:,1]
        meff = MEFF[1]

    lamb = meff
    #lamb = 0

    tot = ((DF[pos] == x) / l_ma).sum()
    return tot / (meff + lamb)


def freq_pair(pos1, pos2, x1, x2):
    if pos2 < LEN_PROT:
        l_ma = MA[:,0]
        meff = MEFF[0]
    elif pos1 >= LEN_PROT:
        l_ma = MA[:,1]
        meff = MEFF[1]
    else:
        l_ma = MA.mean(axis=1)
        meff = sum(MEFF)/2

    lamb = meff

    M, _ = DF.shape

    if pos1 == pos2:
        return freq_one(pos1, x1) * (x1 == x2)




    tot = (((DF[pos1] == x1) & (DF[pos2] == x2)) / l_ma).sum()
    return tot / (meff + lamb)
        

def freq_one_pc(pos, x):
    

    if pos < LEN_PROT:
        q = 21
    else:
        q = 5

    return (1 - PSEUDOCOUNT_WEIGHT) * freq_one(pos, x) + PSEUDOCOUNT_WEIGHT / q 


def freq_pair_pc(pos1, pos2, x1, x2):


    if pos1 < LEN_PROT:
        q = 21
    else:
        q = 5

    if pos1 == pos2:
        return (1 - PSEUDOCOUNT_WEIGHT) * freq_pair(pos1, pos2, x1, x2) + PSEUDOCOUNT_WEIGHT / q * (x1 == x2)

    if pos2 < LEN_PROT:
        q *= 21
    else:
        q*= 5

    return (1 - PSEUDOCOUNT_WEIGHT) * freq_pair(pos1, pos2, x1, x2)  + PSEUDOCOUNT_WEIGHT / q 


def ma(pos, threshold=0.8):
    """We try to efficiently compute the values of ma for each
    sequence"""


    prots = pd.DataFrame([seq[:LEN_PROT] for seq in DF.values])
    rnas = pd.DataFrame([seq[LEN_PROT:] for seq in DF.values])
    prot_seq = prots.iloc[pos,:]
    rna_seq = rnas.iloc[pos,:]

    _, l_prot = prots.shape
    _, l_rna = rnas.shape

    return ((prots.eq(prot_seq).sum(axis=1) >= threshold * l_prot).sum(),
            (rnas.eq(rna_seq).sum(axis=1) >= threshold * l_rna).sum())


def slave_ma(pos, threshold=0.8):
    return pos, ma(pos, threshold)


@lru_cache(maxsize=1)
def get_meff():
    M, _ = DF.shape
    return (sum(2/m.sum() for m in MA[:,0]),
            sum(2/m.sum() for m in MA[:,1]))


def compute_c(p):
    (i, a), (j, b) = p
    
    fo = D_FREQ_ONE


    res = freq_pair_pc(i, j, a, b) - fo[i, a] * fo[j, b]

    return i, j, a, b, res


def slave_freq_one_pc(args):
    """Wrapper to be able to use Pool"""
    return args, freq_one_pc(*args)


def slave_freq_pair_pc(args):
    """Wrapper to be able to use Pool"""
    pos1, x1 = args[0]
    pos2, x2 = args[1]
    return args, freq_pair_pc(pos1, pos2, x1, x2)


def to_do_generator():
    for col in DF.columns:
        if col < LEN_PROT:
            for u in SYMBOLS_PROT[:-1]:
                yield (col, u)
        else:
            for u in SYMBOLS_RNA[:-1]:
                yield (col, u)


def to_do_generator_all():
    for col in DF.columns:
        if col < LEN_PROT:
            for u in SYMBOLS_PROT:
                yield (col, u)
        else:
            for u in SYMBOLS_RNA:
                yield (col, u)


def make_c(tot_to_do):

    col_map = {x:i for i, x in enumerate(to_do_generator())}


    c = np.zeros((len(col_map), len(col_map)),  np.float64)


    to_do = product(to_do_generator(), to_do_generator())


    with Pool(NB_PROCS) as pool:
        for i, j, a, b, res in tqdm(pool.imap_unordered(compute_c, to_do), total=tot_to_do):

            mat_pos_a = col_map[i, a]
            mat_pos_b = col_map[j, b]
            c[mat_pos_a][mat_pos_b] = res
    
            

    return c            


def slave_compute_results(args):
    i, j = args

    W_mf = returnW(i, j, map_key)
    di_mf_pc = bp_link(i, j, W_mf)
    return (i, j), di_mf_pc


def returnW(i, j, map_key):
    if i < LEN_PROT:
        q1 = 21
        alphabet1 = SYMBOLS_PROT
    else:
        q1 = 5
        alphabet1 = SYMBOLS_RNA
    if j < LEN_PROT:
        q2 = 21
        alphabet2 = SYMBOLS_PROT
    else:
        q2 = 5
        alphabet2 = SYMBOLS_RNA

    W = np.ones((q1, q2))
    k1 = map_key[i, alphabet1[0]]
    k2 = map_key[i, alphabet1[-2]] + 1
    l1 = map_key[j, alphabet2[0]]
    l2 = map_key[j, alphabet2[-2]] + 1


    W[:q1 - 1, :q2 -1] = np.exp(-INV_C[k1:k2, l1:l2]) 
    #W[:q1 - 1, :q2 -1] = (-INV_C[k1:k2, l1:l2]) 
    #W[:q1 - 1, :q2 -1] = mpmath.matrix(-INV_C[k1:k2, l1:l2]).apply(mpmath.exp).tolist()

    return W


def bp_link(i, j, W):
    mu1, mu2 = compute_mu(i, j, W)
    di = compute_di(i, j, W, mu1, mu2)
    return di


def compute_mu(i, j, W):
    epsilon = 1e-8
    diff = 1
    if i < LEN_PROT:
        q1 = 21
        pi = [D_FREQ_ONE[i, x] for x in SYMBOLS_PROT]
    else:
        q1 = 5
        pi = [D_FREQ_ONE[i, x] for x in SYMBOLS_RNA]
    if j < LEN_PROT:
        q2 = 21
        pj = [D_FREQ_ONE[j, x] for x in SYMBOLS_PROT]
    else:
        q2 = 5
        pj = [D_FREQ_ONE[j, x] for x in SYMBOLS_RNA]

    mu1 = np.ones((1, q1)) / q1
    mu2 = np.ones((1, q2)) / q2

    while diff > epsilon:

        scra1 = mu2 @ W.transpose()
        scra2 = mu1 @ W;


        
        new1 = np.divide(pi, scra1)
        new1 = new1 / new1.sum()

        new2 = np.divide(pj, scra2)
        new2 = new2 / new2.sum()


        diff = max(abs(new1 - mu1).max(), abs(new2 - mu2).max())

        mu1 = new1
        mu2 = new2

    return mu1, mu2


def compute_di(i, j, W, mu1, mu2):
    tiny = 1e-100


    pdir = np.multiply(W, mu1.transpose() @ mu2)
    pdir = pdir / pdir.sum().sum()

    if i < LEN_PROT:
        alphabet1 = SYMBOLS_PROT
        pi = [D_FREQ_ONE[i, x] for x in SYMBOLS_PROT]
    else:
        alphabet1 = SYMBOLS_RNA
        pi = [D_FREQ_ONE[i, x] for x in SYMBOLS_RNA]
    if j < LEN_PROT:
        alphabet2 = SYMBOLS_PROT
        pj = [D_FREQ_ONE[j, x] for x in SYMBOLS_PROT]
    else:
        alphabet2 = SYMBOLS_RNA
        pj = [D_FREQ_ONE[j, x] for x in SYMBOLS_RNA]

    pi = np.array(pi)
    pj = np.array(pj)


    di = 0
    for pos_a, a in enumerate(alphabet1):
        for pos_b, b in enumerate(alphabet2):
            p = pdir[pos_a, pos_b]
            di += p * np.log((p + tiny) / (D_FREQ_ONE[i, a] * D_FREQ_ONE[j, b] + tiny))
    return di


def pos_gaps(df, gaps):
    """find all positions in alignment with  more than conserved at more than conservation"""
    nb_rows, nb_cols = df.shape

    value_counts = df.apply(pd.Series.value_counts, axis=0)#.max(axis=0).ge(conservation * nb_rows)

    ge = []
    for i in value_counts.columns:
        try:
            if value_counts[i]['-'] > nb_rows * gaps:
                ge.append(i)
                continue
        except:
            pass
        try:
            if value_counts[i]['.'] > nb_rows * gaps:
                ge.append(i)
                continue
        except:
            pass
    return ge


def pos_conserved(df, conservation):
    """find all positions in alignment with  more than conserved at more than conservation"""
    nb_rows, nb_cols = df.shape

    value_counts = df.apply(pd.Series.value_counts, axis=0).max(axis=0).ge(conservation * nb_rows)

    ge = [i for i, x in enumerate(value_counts) if x]
    return ge


def make_results_df(results):
    """transform dict of results in a dataframe"""
    max_val = max(x[1] for x in results)

    df = []
    for i in range(max_val + 1):
        df.append([])
        for j in range(max_val + 1):
            df[-1].append(results.get((i, j), np.nan))
    return pd.DataFrame(df)


def restrain_results_prot_rna(results, len_prot, other=None):
    """retrain the df results to the prot (y-axis) and rna (x-axis)
    if additional positions are given, remove them too
    """
    results.drop(list(range(len_prot)), axis=1, inplace=True)
    max_col = max(results.columns)
    results.drop(list(range(len_prot, max_col + 1)), axis=0, inplace=True)

    if other is not None:
        drop_prot = [x for x in other if x < len_prot]
        drop_rna = [x for x in other if x >= len_prot]
        results.drop(drop_prot, axis=0, inplace=True)
        results.drop(drop_rna, axis=1, inplace=True)


def compute_apc(results):
    apc = results.copy()
    for i in results.index:
        for j in results.columns:
            val = results.loc[i, j]
            vals_row = results.loc[i,:].values
            vals_col = results[j].values
            

            mean_all = mean(y for x in results.values for y in x)
            mean_row = mean(vals_row)
            mean_col = mean(vals_col)

            apc.loc[i, j] = val - mean_row * mean_col / mean_all

    return apc


def get_apc(df, results, len_prot):
    """We compute the APC but after removing
    positions with too many gaps or too high conservation"""


    list_gaps = pos_gaps(df, GAPS)
    list_pos_conserved = pos_conserved(df, CONSERVATION)
    to_remove = list_gaps + list_pos_conserved
 
    results = make_results_df(results)
    restrain_results_prot_rna(results, len_prot, to_remove)


    apc = compute_apc(results)

    return apc


def main(path_input, path_output):
    global DF
    global MEFF
    global MA
    global LEN_PROT
    global LEN_RNA
    global D_FREQ_ONE
    global INV_C


    #Gather the data, len of proteins and RNAs, make a dataframe out of it
    data, LEN_PROT, LEN_RNA = read_merged_fasta(path_input)
    DF = make_df(data)
    M, L = DF.shape


    #First compute the values of MA and MEFF (one for prots and one for rnas)
    MA = np.zeros((M, 2))
    with timer("get meff / ma"):
        with Pool(NB_PROCS) as pool:
            for i, m in tqdm(pool.imap_unordered(slave_ma, range(M)), total=M):
                MA[i][0] = m[0]
                MA[i][1] = m[1]
        MEFF = get_meff()
    
    print(f"MEFF:{MEFF}")

    #We precompute the frequency of single sites because they are used a lot
    l = len(list(to_do_generator_all()))
    with timer("frequency one"):
        with Pool(NB_PROCS) as pool:
            for k, v in tqdm(pool.imap_unordered(slave_freq_one_pc, to_do_generator_all(), chunksize=10_000), total=l):
                D_FREQ_ONE[k] = v


    #Compute the correlation matrix inverse
    l = len(list(to_do_generator()))
    c = make_c(tot_to_do=l*l)
    INV_C = np.linalg.inv(c)


    #Finally the methods for the DCA
    global map_key
    map_key = {x:i for i, x in enumerate(to_do_generator())}
    d = {}
    with Pool(NB_PROCS) as pool:
        for x, val in pool.imap_unordered(slave_compute_results, combinations(DF.columns, 2)):
            d[x] = val


    #The APC values
    apc = get_apc(DF, d, LEN_PROT)

    output = ["#Prot_pos RNA_pos DCA APC"]
    for i, j in product(range(LEN_PROT), range(LEN_RNA)):
        pos_i = i
        pos_j = j + LEN_PROT
        try:
            output.append(f"{i + 1} {j + 1} {d[pos_i, pos_j]} {apc.loc[pos_i, pos_j]}")
        except KeyError:
            #Positions removed because of gaps or conservation
            print("ERROR:", i, j)
            pass

    output = '\n'.join(output)

    if path_output is not None:
        with open(path_output, 'w') as f:
            pickle.dump(output, f)
    else:
        print(output)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DCA between a protein and RNA")

    parser.add_argument('--input', '-i', required=True,
                        help=""""Path to input file, in kind of FASTA format.
                        Name line should start with ">", following line should have two sequences, a protein and RNA, separated by white space. 
                        Lines with sequences must have the same length. Gaps must be dots ("."). """)
    parser.add_argument('--output', '-o', default=None,
                        help="""Output path to return the DCA values. If none is provided they will be printed to stdout""")
    parser.add_argument('--nprocs', '-n', default=1, type=int, 
                        help="""Number of processors to use for computations, default is 1.""")
    parser.add_argument('--gaps', '-g', default=0.5, type=float, 
                       help="""Maximum percentage of gaps to consider columns in alignment""")
    parser.add_argument('--conservation', '-c', default=0.99, type=float,
                        help="""Maximal conservation in a column to consider in alignment""")

    args = parser.parse_args()

    NB_PROCS = args.nprocs
    GAPS = args.gaps
    CONSERVATION = args.conservation

    main(args.input, args.output)
