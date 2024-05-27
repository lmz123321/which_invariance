## Causal Robust Learning

Official implementation for the ICML 2023 paper: [Which Invariance Should We Transfer: A Causal Minimax Learning Approach](https://arxiv.org/abs/2107.01876).

### Requirements

causal-learn, graphical_models, networkx, r-base, ggm, pytorch

To install ggm, activate your R terminal and use ```install.packages('ggm')```

### To reproduce our results 

#### Simulation

Generate synthetic data (or directly use the data provided under ```./data/simulation/```):

```
python ./simulation/data.py
```

Run the training:

```
python ./simulation/main.py
```

Visualization

```
./simulation/draw.ipynb
```

#### ADNI

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

#### IMPC

Preprocess the data (or directly use the data provided under ```./data/impc/```):

```
python ./impc/extract.py
```

Run the training:

```
python ./impc/main.py
```


### To run on custom datasets

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



### Contact

liumingzhou@stu.pku.edu.cn
