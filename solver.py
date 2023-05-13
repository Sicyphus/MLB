import numpy as np , gurobipy as gp
from gurobipy import GRB
from pyomo.environ import *
    
def gurobi(df, m, params, limits, rosters, platform):
    #need to adjust constraints for FD
    # adjust graming for FD (make sure they all have same indices)
    #order constraint not working sometimes
    #SUCCESSive order constraints for when things run out
    # convert dictionary to actual variable names
    p, h, teambool, st = m['p'], m['h'], m['tb'], m['st']
    c, b1, b2, b3, ss, of = m['c'], m['b1'] , m['b2'], m['b3'], m['ss'], m['of']
    B, stack, overlap = params['B'], params['stack'], params['overlap']
    nptch, nhit, no_c, no_1b, no_c1b, no_2b = limits['no_ptch'], limits['no_hit'], limits['no_c'], limits['no_1b'], limits['no_c1b'], limits['no_2b']
    no_3b, no_ss, no_of = limits['no_3b'], limits['no_ss'], limits['no_of']   
     
    pr = np.array(df['Proj FP'])

    cost = np.array(df['Salary'].to_list())  #cost vector
    d = np.array([float(cost[i]) for i in range(len(cost))])

    v1 = np.array(len(df)*[1])  # vector of ones used for summations

    m = gp.Model()
    m.setParam("LogToConsole",0)
    x = m.addMVar(len(df), vtype=GRB.BINARY) # roster 
    v = m.addMVar(len(teambool)*len(st[stack[0]]),vtype=GRB.BINARY)#ensures consecutive
    #w = m.addMVar(len(teambool)*len(st[stack[1]]),vtype=GRB.BINARY)#hitters

    #Setup Constraints
    m.addConstr(d @ x <= B)                                  #budget constraint

    m.addConstr(np.diag(p) @ x @ v1 == nptch)           #  num pitchers
    m.addConstr(np.diag(h) @ x @ v1 == nhit)           #  num hitters

    i = 0  # order counter
    for tbarr in teambool: 
        m.addConstr(stack[0]*(np.diag(tbarr) @ np.diag(p) @ x @ v1) + np.diag(tbarr) @ np.diag(h) @ x @ v1 <= stack[0])    # no opposing pitchers / only a max of mst hitters
        for odx in range(len(st[stack[0]])):          # consecutive batters constraint
           m.addConstr(np.diag(tbarr) @ np.diag(h) @ np.diag(st[stack[0]][odx]) @ x @ v1 >= stack[0]*v[i])
           #m.addConstr(np.diag(tbarr) @ np.diag(h) @ np.diag(st[stack[1]][odx]) @ x @ v1 >= stack[1]*w[i])
           i += 1

    m.addConstr(sum(v) >= 1)    # ensure at least 1 team w order constraint 1
    #m.addConstr(sum(w) >= 1)    # ensure at least 1 team w order constraint 2
    #m.addConstr(v @ w == 0)     # make sure order constraints distinct
           
    m.addConstr(np.diag(c) @ x  @ v1 + np.diag(b1) @ x  @ v1 >= 1)  # c/1b <=2 (FD)
    m.addConstr(np.diag(b2) @ x  @ v1 >= 1)   # positional constraints
    m.addConstr(np.diag(b3) @ x  @ v1 >= 1)   # minimum for DK/FD
    m.addConstr(np.diag(ss) @ x  @ v1 >= 1)    
    m.addConstr(np.diag(of) @ x  @ v1 >= 3) 

    m.addConstr(np.diag(c) @ x  @ v1 <= no_c) 
    m.addConstr(np.diag(b1) @ x  @ v1 <= no_1b)  # positional constraints 
    m.addConstr(np.diag(b2) @ x  @ v1 <= no_2b) 
    m.addConstr(np.diag(b3) @ x  @ v1 <= no_3b) 
    m.addConstr(np.diag(ss) @ x  @ v1 <= no_ss) 
   
    m.addConstr(np.diag(of) @ x  @ v1 <= no_of)   #(below: meant for FD but works w DK
    m.addConstr(np.diag(c) @ x  @ v1 + np.diag(b1) @ x  @ v1 <= no_c1b)  # c/1b <=2 (FD)
    
    for sl in rosters:
        m.addConstr(sl @ x <= overlap)

    m.setObjective(pr @ x, GRB.MAXIMIZE)        #objective function
    m.optimize()

    try: solution = np.array([int(v.X) for v in m.getVars() if abs(v.obj) > 1e-6])
    except AttributeError: solution = []
    return solution
 
def glp(df, m, params, limits, rosters, platform): # GLP solver (uses executable gplsol)
    #need to adjust constraints for FD
    # adjust graming for FD (make sure they all have same indices)
    #order constraint not working sometimes
    #SUCCESSive order constraints for when things run out
    # convert dictionary to actual variable names
    p, h, teambool, st = m['p'], m['h'], m['tb'], m['st']
    c, b1, b2, b3, ss, of = m['c'], m['b1'] , m['b2'], m['b3'], m['ss'], m['of']
    B, stack, overlap = params['B'], params['stack'], params['overlap']
    nptch, nhit, no_c, no_1b, no_c1b, no_2b = limits['no_ptch'], limits['no_hit'], limits['no_c'], limits['no_1b'], limits['no_c1b'], limits['no_2b']
    no_3b, no_ss, no_of = limits['no_3b'], limits['no_ss'], limits['no_of']   
     
    pr = np.array(df['Proj FP'])

    cost = np.array(df['Salary'].to_list())  #cost vector
    d = np.array([float(cost[i]) for i in range(len(cost))])

    v1 = np.array(len(df)*[1])  # vector of ones used for summations

    #m = gp.Model()
    #m.setParam("LogToConsole",0)
    #x = m.addMVar(len(df), vtype=GRB.BINARY) # roster 
    #v = m.addMVar(len(teambool)*len(st[stack[0]]),vtype=GRB.BINARY)#ensures consecutive
    #w = m.addMVar(len(teambool)*len(st[stack[1]]),vtype=GRB.BINARY)#hitters

    #Setup Constraints
    #m.addConstr(d @ x <= B)                                  #budget constraint

    #m.addConstr(np.diag(p) @ x @ v1 == nptch)           #  num pitchers
    #m.addConstr(np.diag(h) @ x @ v1 == nhit)           #  num hitters

    i = 0  # order counter
    #for tbarr in teambool: 
    #    m.addConstr(stack[0]*(np.diag(tbarr) @ np.diag(p) @ x @ v1) + np.diag(tbarr) @ np.diag(h) @ x @ v1 <= stack[0])    # no opposing pitchers / only a max of mst hitters
    #    for odx in range(len(st[stack[0]])):          # consecutive batters constraint
    #       m.addConstr(np.diag(tbarr) @ np.diag(h) @ np.diag(st[stack[0]][odx]) @ x @ v1 >= stack[0]*v[i])
           #m.addConstr(np.diag(tbarr) @ np.diag(h) @ np.diag(st[stack[1]][odx]) @ x @ v1 >= stack[1]*w[i])
    #       i += 1

    #m.addConstr(sum(v) >= 1)    # ensure at least 1 team w order constraint 1
    #m.addConstr(sum(w) >= 1)    # ensure at least 1 team w order constraint 2
    #m.addConstr(v @ w == 0)     # make sure order constraints distinct
           
    #m.addConstr(np.diag(c) @ x  @ v1 + np.diag(b1) @ x  @ v1 >= 1)  # c/1b <=2 (FD)
    #m.addConstr(np.diag(b2) @ x  @ v1 >= 1)   # positional constraints
    #m.addConstr(np.diag(b3) @ x  @ v1 >= 1)   # minimum for DK/FD
    #m.addConstr(np.diag(ss) @ x  @ v1 >= 1)    
    #m.addConstr(np.diag(of) @ x  @ v1 >= 3) 

    #m.addConstr(np.diag(c) @ x  @ v1 <= no_c) 
    #m.addConstr(np.diag(b1) @ x  @ v1 <= no_1b)  # positional constraints 
    #m.addConstr(np.diag(b2) @ x  @ v1 <= no_2b) 
    #m.addConstr(np.diag(b3) @ x  @ v1 <= no_3b) 
    #m.addConstr(np.diag(ss) @ x  @ v1 <= no_ss) 
   
    #m.addConstr(np.diag(of) @ x  @ v1 <= no_of)   #(below: meant for FD but works w DK
    #m.addConstr(np.diag(c) @ x  @ v1 + np.diag(b1) @ x  @ v1 <= no_c1b)  # c/1b <=2 (FD)
    
    #for sl in rosters:
    #    m.addConstr(sl @ x <= overlap)

    #m.setObjective(pr @ x, GRB.MAXIMIZE)        #objective function
    #m.optimize()

    model.value = Objective(expr = sum( v[i]*model.x[i] for i in model.ITEMS ),sense = maximize )

    optimizer=SolverFactory('glpk').solve(model)
    #model.display()
    x = [model.x[i].value for i in model.x]
    if None in x: x = []

