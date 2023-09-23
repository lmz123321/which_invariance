# Which Invariance Should We Transfer? A Causal Minimax Learning Approach

<div align="left">

  <a>![Python 3.6+](https://img.shields.io/badge/Python-3.6%2B-brightgreen.svg)</a>
  <a href="https://github.com/Embracing/GFPose/blob/main/LICENSE">![License](https://img.shields.io/github/license/metaopt/torchopt?label=license&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=)</a>
  [![arXiv](https://img.shields.io/badge/arXiv-2107.01876-b31b1b.svg)](https://arxiv.org/pdf/2107.01876.pdf)
</div>


This repo contains the official implementation for the ICML2023 paper: [Which Invariance Should We Transfer: A Causal Minimax Learning Approach](https://arxiv.org/abs/2107.01876).

**Our method uses causal discovery and equivalence classes searching to find the most robust subset for Out-of-Distribution (OOD) generalization.**

By [Mingzhou Liu](https://scholar.google.com/citations?user=W0VTiFoAAAAJ&hl=en), Xiangyu Zheng, [Xinwei Sun](https://sunxinwei0625.github.io/sunxw.github.io/), Fang Fang, and [Yizhou Wang](http://cfcs.pku.edu.cn/people/faculty/yizhouwang/index.htm).

Video introduction of the paper is available [here](https://recorder-v3.slideslive.com/#/share?share=83385&s=13f03e27-bfb1-49f5-833d-6741d3f22ce5).

## Major Requirements

- causal discovery: ```causal-learn, graphical_models, networkx```
- equivalence classes search: ```r-base, ggm```
- optimal subset selection: ```pytorch```

All the above dependencies except the ```ggm``` can be installed via conda and pip.

To install the ```ggm``` package, activate your ```R``` terminal and use ```install.packages('ggm')```



## To reproduce our results

#### Simulation (Sec. 6.1)

Generate synthetic data (or directly use the data provided under ```./data/simulation/```):

```
python ./simulation/data.py
```

Run the training:

```
python ./simulation/main.py
```

Visualization (also applies to below datasets):

```
./simulation/draw.ipynb
```

#### ADNI dataset (Sec. 6.2)

Download the ADNI dataset [here](https://n.neurology.org/content/74/3/201)

Preprocess and partition of heterogeneous environment according to age:

```
python ./adni/extract.py
python ./adni/partition.py
```

Run the training:

```
python ./adni/main.py
```

#### IMPC dataset (Sec. F.2.1)

Preprocess the data (or directly use the data provided under ```./data/impc/```):

```
python ./impc/extract.py
```

Run the training:

```
python ./impc/main.py
```



## To run on custom datasets

#### Step-I: Causal discovery

Command:

```
python ./causal_discovery/causal_discovery.py -p path_to_data_folder -f data_filename 
```

Example:

```
python ./causal_discovery/causal_discovery.py -p ./data/simulation/ -f 134581011151617.csv
```

Explanation:

this command uses the [CD-NOD](https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/CDNOD.html) algorithm for heterogeneous causal discovery, it generates the followings under the ```cache``` folder:

- ```auggraph.gml```: augment causal graph over $\{Y,\mathbf{X}_S,\mathbf{X}_M,E\}$.
- ```stablegraph.gml```: stable causal graph over $\{Y,\mathbf{X}_S\}$.

- ```graphparse.json```: local components such as $\mathbf{X}_M,\mathbf{X}_M^0,\mathbf{W}$, ect.

- ```cit_cache.json```: intermediate results of the conditional independence (CI) test.

#### Step-II: Equivalence classes search

Command:

```
python ./eqcls_search/eqcls_search.py -f stable_dag_gml_file
```

Example:

```
python ./eqcls_search/eqcls_search.py -f ./cache/stablegraph.gml
```

Explanation:

this command implements the Alg.2  in our paper to search for equivalence classes, it generates the followings under the ```cache``` folder:

- ```eqclses.json```: all equivalence classes
- ```eqsubsets.json```: the $N_G$ subsets we need to search

#### Step-III:Optimal subset section

Command:

```
python ./findoptset/main.py -p graphparser_file -e eqsubsets_file -d path_to_data_folder
```

Example:

```
python ./findoptset/main.py -p ./cache/graphparse.json -e ./cache/eqsubsets.json -d ./data/simulation/
```

Explanation:

this command implements Alg.1 in our paper to find the optimal subset (invariant predictor), it generates a ```record.json``` under the ```./findopeset/``` folder, which records the estimated and ground-truth worst-case risk for each subset.



## Citation

```
@inproceedings{liu2023invariance,
  title={Which Invariance Should We Transfer? A Causal Minimax Learning Approach},
  author={Liu, Mingzhou and Zheng, Xiangyu and Sun, Xinwei and Fang, Fang and Wang, Yizhou},
  booktitle={International Conference on Machine Learning},
  pages={},
  year={2023},
  organization={PMLR}
}
```

