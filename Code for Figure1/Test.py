import os
import sys

import numpy as np
import scipy.io as scio
import time
import pygmo as pg

from pyMC import MC_HVC, MC2
from pyR2 import generate_WV_grid, R2HVC

# CMD code: python TestMCR2.py 3
if __name__ == "__main__":
    path = 'DTLZ2-10.mat'
    data = scio.loadmat(path)


    def CIR(output, pred, count=0, criteria='min'):  # [bs, 100, 1]  contain nan
        for x, y in zip(output, pred):  # [100, 1]
            if criteria == 'min':
                count = count + int(np.argmin(x) == np.argmin(y))
            elif criteria == 'max':
                count = count + int(np.argmax(x[~np.isnan(x)]) == np.argmax(y[~np.isnan(y)]))
            else:
                raise Exception('un-implemented CIR criteria. ')

        return count

    ## loading process 
    solutionset = data.get('Data')  # [dataset_num, data_num, M]
    hvc = data.get('HVCval')  # [dataset_num, data_num]
    hvc = np.reshape(hvc, (hvc.shape[0], hvc.shape[1], 1))    # [dataset_num, data_num, 1]

    L, N, M = solutionset.shape

    sampleNum = np.arange(100, 1000, 100)   # [20,]   [5, 10, ..., 100]
    lineNum = np.arange(100, 1000, 100)   # [20,]   [5, 10, ..., 100]

    Time_BF = np.zeros(1)
    CIRmin_BF = np.zeros(1)

    Time_MC = np.zeros(len(sampleNum))
    CIRmin_MC = np.zeros(len(sampleNum))

    Time_R2 = np.zeros(len(lineNum))
    CIRmin_R2 = np.zeros(len(lineNum))

    reference_point = [1.5 for _ in range(M)]
    hv_algo = pg.bf_approx(use_exact=False, eps=1, delta=1)

    # Test RMC method
    for k in range(1):
         print(f'BF {k}')
         CIR_min_count = 0
         CIR_total = 0
         start_time = time.time()
         for i in range(L):
             pop = solutionset[i]        # [N, M]
             hv = pg.hypervolume(pop)
    
             leastindex = hv.least_contributor(reference_point, hv_algo=hv_algo)
    
    
             # CIR_min
             CIR_min_count = CIR_min_count + int(np.argmin(hvc[i][np.newaxis, :]) == leastindex)
    
             CIR_total = CIR_total + 1
    
         CIR_min = CIR_min_count / CIR_total
         end_time = time.time()
    
         CIRmin_BF[k] = CIR_min
         Time_BF[k] = end_time - start_time
         print(f'CIR min {CIR_min}, time {end_time - start_time}')

    # Test MC method (i.e., point-based method)
    for k in range(len(sampleNum)):
        print(f'MC {k}')
        loss = []
        CIR_min_count = 0
        CIR_max_count = 0
        CIR_total = 0
        start_time = time.time()
        for i in range(L):
            pop = solutionset[i]        # [N, M]

            hvcmc = np.array([MC_HVC(pop, index, sampleNum[k], reference_point, is_maximum=False)
                     for index in range(len(pop))])
            hvcmc = hvcmc.reshape(hvcmc.shape[0], 1)


            # CIR_min
            CIR_min_count = CIR(hvc[i][np.newaxis, :], hvcmc[np.newaxis, :], CIR_min_count, criteria='min')

           
            CIR_total = CIR_total + 1

        CIR_min = CIR_min_count / CIR_total
        end_time = time.time()

        CIRmin_MC[k] = CIR_min
        Time_MC[k] = end_time - start_time
        print(f'CIR min {CIR_min}, time {end_time - start_time}')

    # Test R2 method (i.e., line-based method)
    for k in range(len(lineNum)):
        print(f'R2 {k}')
        loss = []
        CIR_min_count = 0
        CIR_max_count = 0
        CIR_total = 0
        vectors = generate_WV_grid(lineNum[k], M)
        start_time = time.time()
        for i in range(L):
            pop = solutionset[i]        # [N, M]

            hvcr2 = np.array([R2HVC(pop, vectors, index, reference_point, is_maximize=False)
                     for index in range(len(pop))])
            hvcr2 = hvcr2.reshape(hvcr2.shape[0], 1)

            # CIR_min
            CIR_min_count = CIR(hvc[i][np.newaxis, :], hvcr2[np.newaxis, :], CIR_min_count, criteria='min')

            
            CIR_total = CIR_total + 1

        CIR_min = CIR_min_count / CIR_total
        end_time = time.time()

        CIRmin_R2[k] = CIR_min
        Time_R2[k] = end_time - start_time
        print(f'CIR min {CIR_min}, time {end_time - start_time}')

    # save
    scio.savemat('result10.mat',{
        'CIRbf': CIRmin_BF,
        'Timebf': Time_BF,
        'CIRmc':CIRmin_MC,
        'Timemc':Time_MC,
        'CIRr2': CIRmin_R2,
        'Timer2': Time_R2
    })
