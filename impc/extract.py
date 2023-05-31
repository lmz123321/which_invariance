import pandas as pd
import os
from itertools import combinations

# This script generate the training data from mouse.csv
# It choose certain domains (given by envids) into the training set

class IMPCgeneData():
    def __init__(self, ):
        super(IMPCgeneData, self).__init__()

        self.csvfilename = './mouse.csv'

        self.varnames = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'Y', 'E']
        self.colnames = ['IMPC_HEM_030_001', 'IMPC_HEM_032_001', 'IMPC_HEM_036_001', 'IMPC_HEM_037_001',
                         'IMPC_HEM_039_001', 'IMPC_HEM_034_001', 'geno']
        self.colTovar = dict(zip(self.colnames, self.varnames))

        # 0 for wild mouse, other values for different gene knockoff
        self.genonames = ['0', '1797_1', '1796_1', '1798_1', '727_1', '1550_1', '1799_1', '3157_1', '3621_1', '3803_1',
                          '3805_1', '3887_1', '4045_1', '4047_1']

        self.read()

    def read(self, ):
        r"""
        Functs: - read into a DataFrame, take volumns and convert their names
        """
        self.alldata = pd.read_csv(self.csvfilename)[self.colnames]
        self.alldata = self.alldata.rename(columns=self.colTovar)

    def generate(self, envids=[1, 2, 3, 4]):
        r"""
        Functs: - read data from csv file and output a dataFrame like below:
                -   X_1, X_2, X_3, X_4, X_5, Y, E

        Args:   - envids, a list, decide which gene knockoff types the output dataFrame contains
                - e.g. envids = [0] - wild
                       envids = [0,2] - wild and 1796_1

                - the value of E takes 1,2,...,len(envids), denote this row has envids[0],[1],... type of knockoff
        """
        envnames = [self.genonames[ind] for ind in envids]
        output = self.alldata.loc[self.alldata['E'].isin(envnames)].copy(deep=True)

        # map env names into index number 1,2,...,len(envids)
        geneToindex = dict(zip(envnames, [str(i + 1) for i in range(len(envnames))]))
        output['E'] = output['E'].map(geneToindex)

        return output



if __name__ == '__main__':

    impcgene_data = IMPCgeneData()
    OUTBASE = '../data/impc/'

    os.makedirs(OUTBASE, exist_ok=True)

    # generate a data containing all 13+1 domains
    '''
    envids = [i for i in range(14)]
    output = impcgene_data.generate(envids=envids).set_index('X_1')
    output.to_csv(os.path.join(OUTBASE, '{}.csv'.format(''.join(str(e) for e in envids))))
    '''

    # generate all Combination of envids
    # we want, in every training split, there are envid=0 and 3 other random envids (chosen from 1,2,...,13)
    # so, there are C3,13=286 kinds of different combinations
    '''
    envids_allcomb = list(combinations([i + 1 for i in range(13)], 3))
    for envids in envids_allcomb:
        envids = [0] + list(envids)
        output = impcgene_data.generate(envids=envids).set_index('X_1')
        output.to_csv(os.path.join(OUTBASE, '{}.csv'.format(''.join(str(e) for e in envids))))
    '''

    # generate specific training split
    # for example, 0268+XX where XX are C2,10 combinations from rand_ids = [1,3,4,5,7,9,10,11,12,13]
    fixed_ids = [0,2,4,7]

    rand_ids = []
    for i in range(13):
        if (i + 1) not in fixed_ids:
            rand_ids.append(i + 1)

    envids_allcomb = list(combinations(rand_ids, 2))
    for envids in envids_allcomb:
        envids = fixed_ids + list(envids)
        output = impcgene_data.generate(envids=envids).set_index('X_1')
        output.to_csv(os.path.join(OUTBASE, '{}.csv'.format(''.join(str(e) for e in envids))))

        # generate test data for a given training split
        # for example, if 0,2,4,7,10,11 has been chosen for training, then generate 8 test sets from [1,3,5,6,8,9,12,13]
        testids = [i+1 for i in range(13) if i+1 not in envids ]
        os.makedirs(os.path.join(OUTBASE,'{}'.format(''.join(str(e) for e in envids))), exist_ok=True)

        for testid in testids:
            output = impcgene_data.generate(envids=[testid]).set_index('X_1')
            output.to_csv(os.path.join(OUTBASE,'{}'.format(''.join(str(e) for e in envids)), '{}.csv'.format(testid)))