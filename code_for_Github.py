#python code_for_Github.py  4_26 9 4 3 # gets data for 4_26, 9 games, at most 4-stack, budget 50K
import pandas as pd, numpy as np, sys, random, scipy.sparse as sp, gurobipy as gp
from gurobipy import GRB

def col_to_npbool(dframe, col, val):
    vec = np.array((dframe[col] == val).astype(int).to_list())
    return vec

def grader(df, sol_matrix, basefilename):  # score results\output to file
    filename = 'results/{}.csv'.format('_'.join(basefilename))
    scorecum = 0
    with open(filename, 'w') as f:
        for sol in sol_matrix :
            score = df['Actual FP'].dot(sol)
            scorecum += score
            f.write(str(score)+','Z)
        f.write(str(scorecum))

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
    df['Pos'] = df['Pos'].str.split('/').str[0]  # for multi-pos players choose 1st pos
    df = df[df['Proj FP']>0]       # 0 is a bad fantasy score to use

    teamlist = team_sampler(df, int(numg))  # only choose teams from randomly chosen games
    df = df[df['Team'].isin(teamlist)]

    return df, teamlist

def mask_maker_dk(df, teamlist):
    sp = col_to_npbool(df,'Pos','SP')   # pitcher mask
    rp = col_to_npbool(df,'Pos','RP')
    p = np.array([sp[i]+rp[i] for i in range(len(sp))])

    h = np.array([1-p[i] for i in range(len(p))])  # hitter mask

    c = col_to_npbool(df,'Pos','C')
    b1 = col_to_npbool(df,'Pos','1B')   # infielder mask
    b2 = col_to_npbool(df,'Pos','2B')   
    b3 = col_to_npbool(df,'Pos','3B')   
    ss = col_to_npbool(df,'Pos','SS')   

    of = col_to_npbool(df,'Pos','OF')   # outfielder mask

    tb = []               # team masks
    for team in teamlist:     
        tb.append(col_to_npbool(df,'Team',team))

    #masks in dict for easier transport 
    m = {'p': p,'h': h,'c': c,'b1': b1,'b2':b2,'b3':b3,'ss':ss,'of':of,'tb':tb}     
    return m
    
def solver(df, m, params, limits, rosters, platform):
    #need to adjust constraints for FD
    #catcher/1b constraint (fd only)
    #order constraint
    # convert dictionary to actual variable names
    p, h, teambool = m['p'], m['h'], m['tb']
    c, b1, b2, b3, ss, of = m['c'], m['b1'] , m['b2'], m['b3'], m['ss'], m['of']
    B, mst, overlap = params['B'], params['maxst'], params['overlap']
    nptch, nhit, no_c, no_1b, no_2b = limits['no_ptch'], limits['no_hit'], limits['no_c'], limits['no_1b'], limits['no_2b']
    no_3b, no_ss, no_of = limits['no_3b'], limits['no_ss'], limits['no_of']   
    # = no_c + no_1b + no_2b + no_3b + no_ss + no_of
     
    pr = np.array(df['Proj FP'])

    cost = np.array(df['Salary'].to_list())  #cost vector
    d = np.array([float(cost[i]) for i in range(len(cost))])

    v1 = np.array(len(df)*[1])  # vector of ones used for summations

    m = gp.Model()
    x = m.addMVar(len(df), vtype=GRB.BINARY) # roster 


    #Setup Constraints
    m.addConstr(d @ x <= B)                                  #budget constraint

    m.addConstr(np.diag(p) @ x @ v1 == nptch)           #  num pitchers
    m.addConstr(np.diag(h) @ x @ v1 == nhit)           #  num hitters

    for tbarr in teambool:   # no opposing pitchers / only a max of mst hitters 
        m.addConstr(mst*(np.diag(tbarr) @ np.diag(p) @ x @ v1) + np.diag(tbarr) @ np.diag(h) @ x @ v1 <= mst) 
    
    m.addConstr(np.diag(c) @ x  @ v1 <= no_c)  # positional constraints 
    m.addConstr(np.diag(b1) @ x  @ v1 <= no_1b)  # positional constraints 
    m.addConstr(np.diag(b2) @ x  @ v1 <= no_2b) 
    m.addConstr(np.diag(b3) @ x  @ v1 <= no_3b) 
    m.addConstr(np.diag(ss) @ x  @ v1 <= no_ss) 
    m.addConstr(np.diag(of) @ x  @ v1 <= no_of) 
    
    for sl in rosters:
        m.addConstr(sl @ x <= overlap)

    m.setObjective(pr @ x, GRB.MAXIMIZE)        #objective function
    m.optimize()

    solution = np.array([int(v.X) for v in m.getVars() if abs(v.obj) > 1e-6])
    return solution
 
def main():
    date, no_games, max_stck, overlap = sys.argv[1:]
    rosters = []
    params = {'B': float(50000), 'maxst': int(max_stck), 'overlap': int(overlap)}
    limits = {'no_ptch': 2, 'no_hit': 8, 'no_c': 1, 'no_1b': 1, 'no_2b': 1, 'no_3b': 1, 'no_ss': 1, 'no_of': 3}                      
    no_rosters = {'DK':150, 'FD':150}    # 150 DK rosters, #150 FD rosters
    frame, teams = frame_maker_dk(date, no_games)
    masks = mask_maker_dk(frame, teams)
    for i in range(no_rosters['DK']):  #get DraftKings rosters
        soln=solver(frame, masks, params, limits, rosters, 'DK')
        rosters.append(soln)
        
    params['B'] = float(35000) 
    limits = {'no_ptch': 1, 'no_hit': 8, 'no_c': 2, 'no_1b': 2, 'no_2b': 2, 'no_3b': 2, 'no_ss': 2, 'no_of': 4} 
    
    for i in range(no_rosters['FD']): #get FanDuel rosters
        soln=solver(frame, masks, params, limits, rosters,'FD')
        rosters.append(soln)
        
    grader(frame, rosters, sys.argv[2:])#,
    
if __name__ == "__main__":
    main()
