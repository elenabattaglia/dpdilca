import sys
import os
import numpy as np
import logging as l
import datetime


def CreateOutputFile(partial_name, own_directory = False, date = True, overwrite = False):
    '''
    Create and open a file containing the header described below.

    Parameters:
    ----------
    partial_name: partial name of the file and the directory that will contain the file.
    own_directory: boolean. Default: False.
        If true, a new directory './output/_{partial_name}/aaaa-mm-gg_hh.mm.ss' will be created.
        If flase, the path of the file will be './output/_{partial_name}'.
    date: boolean. Default: True.
        If true, the file name will include datetime.
        If false, it will not.
                

    Output
    ------
    f: file (open). Each record contains the following fields, separated by commas (csv file):
        - dilca: variant of Dilca, one of {"M", "RR"}
        - dp_method: differentially private variant of dilca, one of {'su', 'cm'}
        - eps: overall epsilon
        - sigma: sigma parameter for dilca_M
        - h: portion of eps to be used for context computation (only for dp_method == 'su')
        - i: index of the target variable
        - n_values: number of distinct values of variable i
        - pearson: pearson similarity of the distance matrices
        - l1_dist: l1 norm distance of the distance matrices
        - context: context of variable i
        - context_dp: differentially private context of variable i
        - jaccard: jaccard similarity index between context and context_dp
        - overlap: overlap score between context and context_dp
        - date: date of the test
        
        File name:{partial_name}_aaaa-mm-gg_hh.mm.ss.csv or {partial_name}_results.csv
    dt: datetime (as in the directory/ file name)

    
    '''

    
    dt = f"{datetime.datetime.now()}"
    if own_directory:
        data_path = f"./output/_{partial_name}/" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + "/"
    else:
        data_path = f"./output/_{partial_name}/"
    directory = os.path.dirname(data_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    new = True
    if date:
        file_name = partial_name + "_" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + ".csv"
    else:
        file_name = partial_name + '_results.csv'
        if os.path.isfile(data_path + file_name):
            if overwrite:
                os.remove(data_path + file_name)
            else:
                new = False
            
            
    f = open(data_path + file_name, "a",1)
    if new:
        f.write("dilca, dp_method, eps, sigma, h, i, n_values, pearson, l1_dist, context, context_dp, jaccard, overlap, date\n")

    return f, dt



def CreateLogger(input_level = 'INFO'):
    level = {'DEBUG':l.DEBUG, 'INFO':l.INFO, 'WARNING':l.WARNING, 'ERROR':l.ERROR, 'CRITICAL':l.CRITICAL}
    logger = l.getLogger()
    logger.setLevel(level[input_level])
