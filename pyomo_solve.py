from pyomo.environ import *
import pandas as pd
overlap = 5
df = pd.read_csv('data/DFN MLB Hitters FD 4_28.csv')
df_dict = df.set_index('Player Name').to_dict()
rosters = [len(df)*[1],len(df)*[0]]
v = df_dict['Proj FP']
c = df_dict['Salary']
B = 50000
model = ConcreteModel()
model.ITEMS = df['Player Name'].to_list()
model.x = Var( model.ITEMS, within=Binary )

for sl in rosters:
    print(1111)
    rl = dict(zip(df['Player Name'],sl))
    model.overlap = Constraint(expr =  )
def roster_overlap(model, i):
    return sum( rl[i-1]*model.x[i] for i in model.ITEMS ) <= overlap

model.Co1 = pyo.Constraint(rule=Co1)

model.budget = Constraint(expr = sum( c[i]*model.x[i] for i in model.ITEMS ) <= B )

model.value = Objective(
expr = sum( v[i]*model.x[i] for i in model.ITEMS ),
sense = maximize )
optimizer=SolverFactory('glpk').solve(model)
#model.display()
x = [model.x[i].value for i in model.x]
if None in x: x = []



'''
#THIS ONE WORKS!!!!!

from pyomo.environ import *
v = {'hammer':8, 'wrench':3, 'screwdriver':6, 'towel':11}
w = {'hammer':5, 'wrench':7, 'screwdriver':4, 'towel':3}
W_max = 14
model = ConcreteModel()
model.ITEMS = v.keys()
model.x = Var( model.ITEMS, within=Binary )
model.value = Objective(
expr = sum( v[i]*model.x[i] for i in model.ITEMS ),
sense = maximize )
model.weight = Constraint(
expr = sum( w[i]*model.x[i] for i in model.ITEMS ) <= W_max )
optimizer=SolverFactory('glpk')
optimizer.solve(model)
model.display()
'''
'''
from pyomo.environ import *
import pandas as pd, numpy as np
data = {'Player Name': ['Player 3','Player 2','Player 1','Player 0'], 'Salary': [50, 20, 3,51]}
df=pd.DataFrame.from_dict(data)
c=np.array(df['Salary'])
print(c)
model = ConcreteModel()
model.x = Var(df['Player Name'],within=Binary)
model.maximize2 = Objective(expr=sum(model.x[i]*c[i] for i in range(len(model.x))), sense=maximize)
model.Constraint1 = Constraint(expr=model.x<= 100)
optimizer=SolverFactory('glpk')
optimizer.solve(model)
model.display()
'''

'''
from __future__ import division
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
 
## IMPORT DATA ##
fileName = "ff_data.csv"
df = pd.read_csv(fileName)
 
## SETTINGS ##
max_salary = 50000
 
## DATA PREP ##
POSITIONS = ['QB', 'RB', 'WR', 'TE', 'D', 'K']
psn_limits = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 2, 'D': 1, 'K': 1}
PLAYERS = list(set(df['name']))
proj = df.set_index(['name'])['proj_pts'].to_dict()
cost = df.set_index(['name'])['cost'].to_dict()
pos = df.set_index(['name'])['position'].to_dict()
 
## DEFINE MODEL ##
model = ConcreteModel()
 
# decision variable
model.x = Var(PLAYERS, domain=Boolean, initialize=0)
 
# constraint: salary cap
def constraint_cap_rule(model):
 salary = sum(model.x[p] * cost[p] for p in PLAYERS)
 return salary <= max_salary
 
model.constraint_cap = Constraint(rule=constraint_cap_rule)
 
## constraint: positional limits
#def constraint_position_rule(model, psn):
# psn_count = sum(model.x[p] for p in PLAYERS if pos[p] == psn)
# return psn_count == psn_limits[psn]
 
#model.constraint_position = Constraint(POSITIONS, rule=constraint_position_rule)
 
# objective function: maximize projected points
def obj_expression(model):
 return summation(model.x, proj, index=PLAYERS)

model.OBJ = Objective(rule=obj_expression(model), sense=maximize)
 
# good for debugging the model
#model.pprint()
 
## SOLVE ##
opt = SolverFactory('glpk')
opt.solve(instance) 
if results.solver.status:
    model.pprint()
'''

'''    
# create model instance, solve
instance = model.create()
results = opt.solve(instance)
instance.load(results) #load results back into model framework
 
## REPORT ##
print("status=" + str(results.Solution.Status))
print("solution=" + str(results.Solution.Objective.x1.value) + "\n")
print("*******optimal roster********")
P = [p for p in PLAYERS if instance.x[p].value==1]
for p in P:
 print(p + "\t" + pos[p] + "\t cost=" + str(cost[p]) + "\t projection=" + str(proj[p]))
print("roster cost=" + str(sum(cost[p] for p in P)))
'''

'''


import pyomo.environ as pyo
from pyomo.opt import SolverFactory
model = pyo.ConcreteModel()
model.nVars = pyo.Param(initialize=4)
model.N = pyo.RangeSet(model.nVars)
model.x = pyo.Var(model.N, within=pyo.Binary)
model.obj = pyo.Objective(expr=pyo.summation(model.x))
model.cuts = pyo.ConstraintList()
opt = SolverFactory('glpk')
opt.solve(model)

from pyomo.environ import *

# create a model
model = ConcreteModel()

# declare decision variables
model.x = Var(domain=NonNegativeReals)
model.y = Var(domain=NonNegativeReals)

# declare objective
model.profit = Objective(expr = 40*model.x + 30*model.y, sense=maximize)

# declare constraints
model.demand = Constraint(expr = model.x <= 40)
model.laborA = Constraint(expr = model.x + model.y <= 80)
model.laborB = Constraint(expr = 2*model.x + model.y <= 100)

# solve
results = SolverFactory('glpk').solve(model)
results.write()
if results.solver.status:
    model.pprint()

# display solution
print('\nProfit = ', model.profit())

print('\nDecision Variables')
print('x = ', model.x())
print('y = ', model.y())

print('\nConstraints')
print('Demand  = ', model.demand())
print('Labor A = ', model.laborA())
print('Labor B = ', model.laborB())



import pyomo.environ as pyo

m = pyo.ConcreteModel('example')

m.I = pyo.Set(initialize=[1, 2, 3])
m.b = pyo.Var(m.I, domain=pyo.Binary)

# some nonsense objective...
m.obj = pyo.Objective(expr=sum(m.b[i] for i in m.I))

m.pprint()
'''
