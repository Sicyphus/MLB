#python code_for_Github.py  4_26 9 4 3 50000 # gets data for 4_26, 9 games, at most 4-stack, budget 50K
import pandas as pd, numpy as np, sys, random, scipy.sparse as sp, gurobipy as gp
from gurobipy import GRB

def col_to_npbool(dframe, col, val):
    vec = np.array((dframe[col] == val).astype(int).to_list())
    return vec

def team_sampler(dframe, n): # get sample of n teams from team list
    allteams = set(dframe['Team'].to_list())
    if len(allteams)/2 < n:
        print('Not enough games for total teams')
        sys.exit()
    selecteams = random.sample(allteams, n)  # select n teams
    opps = []
    for team in selecteams:  # for each team, find opponent and add to list
        opp = dframe[dframe['Team'] == team]['Opp'].to_list()
        if len(set(opp)) > 1: print('Warning: Opponent team ambiguity')
        else: opps.append(opp[0].replace('@',''))
    return selecteams+opps

def create_base(nparr):  # create basis vector from positions
    tmplst = []
    for elmt in nparr:
        tmplst.append(2**elmt>>1)
    return tmplst
    
def frame_maker_dk(date, numg):
    pdf = pd.read_csv('data/DFN MLB Pitchers DK {0}.csv'.format(date))
    hdf = pd.read_csv('data/DFN MLB Hitters DK {0}.csv'.format(date))

    pdf = pdf[['Player Name','Pos','Salary','Team','Opp','Proj FP','Actual FP']]
    hdf = hdf[['Player Name','Pos','Salary','Team','Opp','Batting Order (Confirmed)','Proj FP','Actual FP']]
    hdf = hdf[hdf['Batting Order (Confirmed)'] != 'x']    #only confirmed ordered batters

    df = pd.concat([pdf, hdf])
    df = df[df['Proj FP']>0]       # 0 is a bad fantasy score to use

    teamlist = team_sampler(df, int(numg))  # only choose teams from randomly chosen games
    df = df[df['Team'].isin(teamlist)]

    return df, teamlist

def mask_maker_dk(df, teamlist):
    sp = col_to_npbool(df,'Pos','SP')   # pitcher mask
    rp = col_to_npbool(df,'Pos','RP')
    p = np.array([sp[i]+rp[i] for i in range(len(sp))])

    h = np.array([1-p[i] for i in range(len(p))])  # hitter mask

    b1 = col_to_npbool(df,'Pos','1B')   # infielder mask
    b2 = col_to_npbool(df,'Pos','2B')   
    b3 = col_to_npbool(df,'Pos','3B')   
    ss = col_to_npbool(df,'Pos','SS')   

    of = col_to_npbool(df,'Pos','OF')   # outfielder mask

    tb = []               # team masks
    for team in teamlist:     
        tb.append(col_to_npbool(df,'Team',team))

    m = {'p': p, 'h': h, 'b1': b1, 'b2':b2, 'b3':b3, 'ss':ss, 'of':of, 'tb':tb}#masks in dict for easier transport 
     
    return m
    
def solver(df, masks, params, limits, rosters):
    # convert dictionary to actual variable names
    for k in masks: exec('{KEY} = {VALUE}'.format(KEY = k, VALUE = repr(masks[k])))
    for k in params: exec('{KEY} = {VALUE}'.format(KEY = k, VALUE = repr(params[k])))
    for k in limits: exec('{KEY} = {VALUE}'.format(KEY = k, VALUE = repr(limits[k])))
    
    pr = np.array(df['Proj FP'])

    cost = np.array(df['Salary'].to_list())  #cost vector
    c = np.array([float(cost[i]) for i in range(len(cost))])

    v1 = np.array(len(df)*[1])  # vector of ones used for summations

    m = gp.Model()
    x = m.addMVar(len(df), vtype=GRB.BINARY) # roster 


    #Setup Constraints
    m.addConstr(c @ x <= B)                                  #budget constraint

    m.addConstr(np.diag(p) @ x @ v1 == nptch)           #  2 pitchers
    m.addConstr(np.diag(h) @ x @ v1 == nhit)           #  7 hitters

    for tbarr in teambool:   # no opposing pitchers 
        m.addConstr(mst*(np.diag(tbarr) @ np.diag(p) @ x @ v1) + np.diag(tbarr) @ np.diag(h) @ x @ v1 <= mst) 
    
    m.addConstr(np.diag(b1) @ x  @ v1 == no_1b)  # positional constraints 
    m.addConstr(np.diag(b2) @ x  @ v1 == no_2b) 
    m.addConstr(np.diag(b3) @ x  @ v1 == no_3b) 
    m.addConstr(np.diag(ss) @ x  @ v1 == no_ss) 
    m.addConstr(np.diag(of) @ x  @ v1 == no_of) 
    
    for rosters in sl:
        m.addConstr(sl @ x <= overlap)

    m.setObjective(pr @ x, GRB.MAXIMIZE)        #objective function
    m.optimize()

    solution = np.array([int(v.X) for v in m.getVars() if abs(v.obj) > 1e-6])

 
def main():
    date, no_games, max_stck, overlap, B = sys.argv[1:]
    rosters = []
    params = {'B': float(B), 'maxst': int(max_stck), 'overlap': int(overlap)}
    limits = {'no_ptch': 2, 'no_1b': 1, 'no_2b': 1, 'no_3b': 1, 'no_ss': 1, 'no_of': 3}                      
    no_rosters = {'DK':150, 'FD':150}    # 150 DK rosters, #150 FD rosters
    frame, teams = frame_maker_dk(date, no_games)
    masks = mask_maker_dk(frame, teams)
    s=solver(frame, masks, params, limits, rosters)
    rosters.append(s)
    
if __name__ == "__main__":
    main()
