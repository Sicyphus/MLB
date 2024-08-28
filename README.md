INSTALL: sudo apt-get install python3-pip glpk-utils
         python -m pip install pandas, scipy, gurobipy, pyomo
         


Fantasy Baseball IP Code
======================

This is the Python version of the code for the paper [Picking Winners Using Integer Programming](http://arxiv.org/pdf/1604.01455v2.pdf) by [David Hunter](http://orc.scripts.mit.edu/people/student.php?name=dshunter), [Juan Pablo Vielma](http://www.mit.edu/~jvielma/), and [Tauhid Zaman](http://zlisto.scripts.mit.edu/home/). The original code was in Julia.  Below are details on the required software, and instructions on how to run different variations of the code. 

## Required Software 
- [GLPK](https://www.gnu.org/software/glpk/)
- [Gurobi](https://support.gurobi.com)
- [Python3](https://www.python.org)
- [Pandas](https://pandas.pydata.org)
- [Numpy](https://numpy.org)


As we noted in the paper, [GLPK](https://www.gnu.org/software/glpk/) and [Cbc](https://projects.coin-or.org/Cbc) are both open-source solvers. This code uses GLPK because we found that it was slightly faster in practice for the formulations considered. For those that want to build more sophisticated models, they can buy [Gurobi](http://www.gurobi.com/). Please consult the [JuMP homepage](https://support.gurobi.com) for details on how to use different solvers. JuMP makes it easy to change between a number of open-source and commercial solvers. 



## Downloading the Code 

You can download the code and the example csv files by calling 

```
$ git clone https://github.com/Sicyphus/MLB.git
```

Alternatively, you can download the zip file from above. 



## Running the Code
python code_for_Github.py 6_4 9 3   


Means for the April 4 slate of games, choose nine games, and find the 150 optimal lineups for an overlap of 3.  The algorithm first uses the 5/3 stack formulation: assume there are five players from one MLB team and 3 from another (with the last player coming from one of the non-used squads).  When those possibilities have been exhausted, use the 4/4 formulation and so forth.  Of course there are limits to overlap and stacking, depending on the number of games/players.

The file experiment.py generates several results for different num. games/overlap combinations.  This will allow you to compare results. The file solve.py sets up the matrices for feeding to the optimization algorithm.

