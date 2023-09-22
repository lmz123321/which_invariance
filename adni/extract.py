# split the Frontal, Temporal, Parietal, Occipital, Cingulum regions, into sup,mid,inf
# generate a 25 nodes version of ADNI causal graph
# B1-B22, C1-C3, S4, E=Age
# aggregate those variables from ADNI30T15T_all and output a aggrADNI30T15TAge25Nodes.csv

import pandas as pd
import os
import scipy.io as scio
import numpy as np
from collections import Counter


def delete_digit(string):
    """
    Functs: - delete digits at the begining and the ending of AAL biomarker names
            - e.g. string='108 Cerebelum_10_R 9082\n'
            - return 'AAL_9082'
    """
    out = []
    llen = len(string)
    for ind in range(llen):
        i = string[ind]

        if ind <= 4 and i.isdigit():
            continue
        # elif ind >= llen-5 and i.isdigit():
        #    continue
        elif i == ' ' or i == '\n' or i == '':
            continue
        else:
            out.append(i)

    return 'AAL_' + ''.join(out[-4:])


def list_add(aa, bb):
    r"""
    Functs: - element wise list adding
    """
    assert not (len(aa) == 0 and len(bb) == 0)
    if len(aa) == 0:
        aa = [0 for b in bb]
    if len(bb) == 0:
        bb = [0 for a in aa]

    ssum = [sum(x) for x in zip(aa, bb)]

    return ssum


class ADNIData():
    def __init__(self):
        super(ADNIData, self).__init__()

        # use which variable to divide domains
        self.env = 'Age'

        # need which brain regions, demos, scalograms
        self.bregions = ['B{}'.format(ind + 1) for ind in range(22)]
        self.covariates = ['Sex', 'Edu', 'NAPOE4']
        self.scalograms = ['ADAS', 'ANARTERR', 'BOST', 'FAQ', 'MMSE', 'NPIQ', 'TR']

        # reading file paths
        self.BASE = '/data/liumingzhou/ADNI/feature_{}_BL/{}'
        self.MODS = ['30T', '15T']
        self.DIAGS = ['AD', 'MCI', 'NC']

        # structure
        self.aggrData = {'Data_id': [], '{}'.format(self.env): []}
        for covariate in self.covariates:
            self.aggrData[covariate] = list()
        for scalogram in self.scalograms:
            self.aggrData[scalogram] = list()
        for bregion in self.bregions:
            self.aggrData[bregion] = list()

        # this is used to map ScanY
        filename = '/data/liumingzhou/ADNI/MRILIST_2020.csv'
        self.IdLists = pd.read_csv(filename)

        # this is used to map B_i
        filename = '/data/liumingzhou/ADNI/aal.txt'
        with open(filename, 'r') as f_in:
            lines = f_in.readlines()
            f_in.close()

        self.aal_indices = [delete_digit(biom) for biom in lines if biom != '\n']

        for aal_index in self.aal_indices:
            self.aggrData[aal_index] = []

    def read(self, ):
        r"""
        Functs: - read raw data from mat files
                - into aggrData
        """
        for MOD in self.MODS:
            for DIAG in self.DIAGS:
                BASE = self.BASE.format(MOD, DIAG)
                # 1. Demographics: Data_id,Age,Sex,Edu,NAPOE4
                filename = 'Label/DEMO.mat'
                demo = scio.loadmat(os.path.join(BASE, filename))['DEMO']

                sex = list(demo[:, 0])
                age = list(demo[:, 1])
                edu = list(demo[:, 2])
                data_id = ['{}_{}_{}'.format(MOD, DIAG, str(i).zfill(6)) for i in range(len(sex))]

                self.aggrData['Data_id'] += data_id
                self.aggrData['Age'] += age
                self.aggrData['Sex'] += sex
                self.aggrData['Edu'] += edu

                filename = 'Label/APOE_CSS.mat'
                data = scio.loadmat(os.path.join(BASE, filename))['APOE_CSS']
                apoe = [list(i).count(4) if list(i) != [-100, -100] else float('nan') for i in data]

                self.aggrData['NAPOE4'] += apoe

                # 2. Scalograms: remove invalid data points
                for scalogram in self.scalograms:
                    data = scio.loadmat(os.path.join(BASE, 'Label', scalogram + '.mat'))[scalogram][:, 0].tolist()
                    data_valid = list()
                    for da in data:
                        if da == -100:
                            data_valid.append(float('nan'))
                        elif da < 0:
                            data_valid.append(abs(da))
                        else:
                            data_valid.append(da)
                    self.aggrData[scalogram] += data_valid

                # 4. AAL volumns
                filename = 'AAL/AAL_{}_feature_TIV.mat'.format(DIAG)
                data = scio.loadmat(os.path.join(BASE, filename))['feature_TIV']

                biomarker_vols = [list(d) for d in data.T]
                bioData = dict(zip(self.aal_indices, biomarker_vols))

                for aal_index in self.aal_indices:
                    self.aggrData[aal_index] += bioData[aal_index]

    def map_aal_regions(self, ):
        r"""
        Functs: - map AAL brain regions to SuStaIn regions and calculate summation of volumes
        Mapping Tab:
                - B1,Frontal Sup Lobe [2101,2102,2111,2112,2601,2602]
                - B2,Frontal Mid Lobe [2201,2202,2211,2212,2611,2612]
                - B3,Frontal Inf Lobe [2301,2302,2311,2312,2321,2322]
                - B4,Temporal Sup Lobe [8111,8112]
                - B5,Temporal Mid Lobe [8201,8202]
                - B6,Temporal Inf Lobe [8301,8302]
                - B7,Temporal Pole [8121,8122,8211,8212]
                - B8,Parietal Sup [6101,6102]
                - B9,Parietal Inf [6201,6202]
                - B10,Occipital Sup [5101,5102]
                - B11,Occipital Mid [5201,5202]
                - B12,Occipital Inf [5301,5302]
                - B13,Cingulum Ant [4001,4002]
                - B14,Cingulum Mid [4011,4012]
                - B15,Cingulum Post [4021,4022]
                - B16,Insula [3001,3002]
                - Accumbens NaN in AAL version_0
                - B17,Amygdala [4201,4202]
                - B18,Caudate [7001,7002]
                - B19,Hippocampus [4101,4102]
                - B20,Pallidum [7021,7022]
                - B21,Putamen [7011,7012]
                - B22,Thalamus [7101,7102]
        Checking:
                - df.loc[df.Data_id == '15T_NC_000045',['AAL_2102']]
        """
        AAL_regions = [
            [2101, 2102, 2111, 2112, 2601, 2602],
            [2201, 2202, 2211, 2212, 2611, 2612],
            [2301, 2302, 2311, 2312, 2321, 2322],
            [8111, 8112],
            [8201, 8202],
            [8301, 8302],
            [8121, 8122, 8211, 8212],
            [6101, 6102],
            [6201, 6202],
            [5101, 5102],
            [5201, 5202],
            [5301, 5302],
            [4001, 4002],
            [4011, 4012],
            [4021, 4022],
            [3001, 3002],
            [4201, 4202],
            [7001, 7002],
            [4101, 4102],
            [7021, 7022],
            [7011, 7012],
            [7101, 7102]
        ]

        AALtoSuSt = dict(zip(self.bregions, AAL_regions))

        for key in AALtoSuSt.keys():
            aal_regions = AALtoSuSt[key]
            SuSt_vols = []
            for aal_region in aal_regions:
                SuSt_vols = list_add(SuSt_vols, self.aggrData['AAL_{}'.format(aal_region)])

            self.aggrData[key] = SuSt_vols

    def map_discovery(self, filename):
        r"""
        Functs: - create a disData dict()
                - it contains 4 kinds of keys
                1. E (from ScanY)
                2. B (from B2,B5,B7,B9)
                3. C (from Age,Sex,Edu,NAPOE4)
                4. S (from 'ADAS','ANARTERR','BOST','FAQ','MMSE','NPIQ','TR')
                - remove those with -4,-100,NaN values
                1. ScanY  NaN
                2. NAPOE4 NaN
                3. ADAS     -4,-100
                   ANARTERR -100
                   BOST     -100
                   FAQ      -1,-100
                   NPIQ     -100
                   TR       -100
                - finally, map ScanY into index
        """

        self.disData = dict()
        self.disData['Data_id'] = self.aggrData['Data_id']
        for bregion in self.bregions:
            self.disData[bregion] = self.aggrData[bregion]

        for ind, covariate in enumerate(self.covariates):
            self.disData['C{}'.format(ind + 1)] = self.aggrData[covariate]

        for ind, scalogram in enumerate(self.scalograms):
            self.disData['S{}'.format(ind + 1)] = self.aggrData[scalogram]

        self.disData['E'] = self.aggrData['Age']

        # remove invalid values (from 785 to 737)
        self.disDF = pd.DataFrame(self.disData).dropna()

        self.disDF.to_csv(filename)


if __name__ == '__main__':
    adnidata = ADNIData()
    adnidata.read()
    adnidata.map_aal_regions()
    adnidata.map_discovery('aggrADNI30T15TAge25Nodes.csv')
