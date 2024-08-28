# solver.py
#
# This sets up the mathematical formulation for the lineup picket.  Using Gurobi (or GLPK),
# generate the solution predicting the roster with the highest score.  Then the next higher,...
################################################################################################ 

import numpy as np , gurobipy as gp, time
from gurobipy import GRB
from pyomo.environ import *
    
 
def dctry(frm, lst): return dict(zip(frm['PLAYERTEAM'],lst))    

def glp(df, m, params, limits, rosters, platform): # GLP solver (uses executable gplsol)
    p, h, tb, ob, st = dctry(df,m['p']), dctry(df,m['h']), m['tb'], m['ob'], m['st']
    c, b1, b2 = dctry(df,m['c']), dctry(df,m['b1']) , dctry(df,m['b2'])
    b3, ss, of = dctry(df,m['b3']), dctry(df,m['ss']), dctry(df,m['of'])
    B, stack, overlap = params['B'], params['stack'], params['overlap']
    nptch, nhit = limits['no_ptch'], limits['no_hit']
    no_c, no_1b, no_c1b, no_2b = limits['no_c'], limits['no_1b'], limits['no_c1b'], limits['no_2b']
    no_3b, no_ss, no_of = limits['no_3b'], limits['no_ss'], limits['no_of']   
    df_dict = df.set_index('PLAYERTEAM').to_dict()    # pyomo likes dictionaries instead of frames

    pr = df_dict['FPTS']

    d = df_dict['SALARY']  #cost vector

    model = ConcreteModel()
    model.ITEMS = df['PLAYERTEAM'].to_list()
    model.ORDERS = range(len(tb)*len(st[stack[0]]))
    model.x = Var( model.ITEMS, within=Binary )
    model.v = Var(model.ORDERS, within=Binary)#ensures consecutive
    model.w = Var(model.ORDERS, within=Binary)#hitters

    #Setup Constraints
    model.budget = Constraint(expr = sum( d[i]*model.x[i] for i in model.ITEMS ) <= B )

    model.pitch = Constraint(expr = sum(p[i]*model.x[i] for i in model.ITEMS) == nptch)           #  num pitchers
    model.hit = Constraint(expr = sum(h[i]*model.x[i] for i in model.ITEMS) == nhit)           #  num hitters

    k = 0  # order counter
    model.teams = ConstraintList(); 
    model.vorder = ConstraintList(); model.worder = ConstraintList(); model.vwcross = ConstraintList()
    model.vwtot = ConstraintList()
    for j in range(len(tb)): 
        tm = dctry(df, tb[j])
        om = dctry(df, ob[j])
        k_ = k    # keep track of first order index for current team
        model.teams.add(sum(tm[i]*h[i]*model.x[i] for i in model.ITEMS) <= stack[0]  )  # only a max of mst hitters
        model.teams.add(sum(stack[0]*tm[i]*p[i]*model.x[i] + om[i]*h[i]*model.x[i] for i in model.ITEMS) <= stack[0]  )  # no opposing pitchers
        for odx in range(len(st[stack[0]])):          # consecutive batters constraint
            sx0 = dctry(df, st[stack[0]][odx])
            sx1 = dctry(df, st[stack[1]][odx])
            model.vorder.add(expr = sum(sx0[i]*tm[i]*h[i]*model.x[i] for i in model.ITEMS) >= stack[0]*model.v[k])
            model.worder.add(expr = sum(sx1[i]*tm[i]*h[i]*model.x[i] for i in model.ITEMS) >= stack[1]*model.w[k])
            k += 1
        model.vwtot.add(expr = sum(model.v[l]+model.w[l]  for l in range(k_,k)) <= 1) # make sure v/w distinct

    model.vsum = Constraint(expr = sum(model.v[i] for i in model.ORDERS) >= 1)    # ensure at least 1 team w order constraint 1
    model.wsum = Constraint(expr = sum(model.w[i] for i in model.ORDERS) >= 1)    # ensure at least 1 team w order constraint 2
           
    model.cb_ = Constraint(expr = sum(c[i]*model.x[i] + b1[i]*model.x[i] for i in model.ITEMS) >= 1)# c/1b <=2 (FD)
    model.b2_ = Constraint(expr = sum(b2[i]*model.x[i] for i in model.ITEMS) >= 1)   # positional constraints
    model.b3_ = Constraint(expr = sum(b3[i]*model.x[i] for i in model.ITEMS) >= 1)   # minimum for DK/FD
    model.ss_ = Constraint(expr = sum(ss[i]*model.x[i] for i in model.ITEMS) >= 1)    
    model.of_ = Constraint(expr = sum(of[i]*model.x[i] for i in model.ITEMS) >= 3) 

    model.c = Constraint(expr = sum(c[i]*model.x[i] for i in model.ITEMS) <= no_c) 
    model.b1 = Constraint(expr = sum(b1[i]*model.x[i] for i in model.ITEMS) <= no_1b)  # positional constraints 
    model.b2 = Constraint(expr = sum(b2[i]*model.x[i] for i in model.ITEMS) <= no_2b) 
    model.b3 = Constraint(expr = sum(b3[i]*model.x[i] for i in model.ITEMS) <= no_3b) 
    model.ss = Constraint(expr = sum(ss[i]*model.x[i] for i in model.ITEMS) <= no_ss) 
   
    model.of = Constraint(expr = sum(of[i]*model.x[i] for i in model.ITEMS) <= no_of)   #(below: meant for FD but works w DK
    model.cb = Constraint(expr = sum(c[i]*model.x[i] + b1[i]*model.x[i] for i in model.ITEMS) <= no_c1b)  # c/1b <=2 (FD)

    model.overlap = ConstraintList()  # ensure not over requested amount overlap 
    for sl in rosters:
        rl = dctry(df, sl)
        model.overlap.add(expr = sum( rl[i]*model.x[i] for i in model.ITEMS ) <= overlap)

    model.value = Objective(expr = sum( pr[i]*model.x[i] for i in model.ITEMS ),sense = maximize )

    curtime = time.time()

    optimizer=SolverFactory('gurobi').solve(model)  # or replace gurobi w/ glpk
    #model.display()
    x = []
    for i in model.x:
        if model.x[i].value == None or time.time()-curtime > 100: return np.array([])
        x.append(int(model.x[i].value) )
    x = np.array(x)
    if None in x: x = []
    
    return x
