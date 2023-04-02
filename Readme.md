A call graph is a control-flow graph, which represents calling relationships between subroutines in a computer program. Here we have tried to predict which 
function or subroutine would call other function or subroutine using Graph Neural Networks. 
Following GNN models are used:
<ul>
  <li> Graph Convolutional Network
  <li> GraphSage
  <li> Graph Isomorphism Network 
</ul>

## Different features used for the task are:
<ul>
  <li> Centrality measures as node information
  <li> Function name, parent class, parent package and structural information
  <li> Temporal information i.e. the sequence in which function call each other
</ul>


## To create enviornment run
`conda install mamba -c conda-forge -y` <br>
`mamba init bash` <br>
`mamba env create -f environment.yaml` <br>

## To activate env
`conda activate callgraphs` <br>

## To see the updated results and recreating the model in your local system
- Setup the Anaconda environment
- Run the individual scripts for the 3 GCN Models

