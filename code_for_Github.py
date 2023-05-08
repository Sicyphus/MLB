#python code_for_Github.py  4_26 9 4 3 # gets data for 4_26, 9 games, at most 4-stack, budget 50K
import pandas as pd, numpy as np, sys, random, scipy.sparse as sp, gurobipy as gp, time
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
            f.write(str(score)+',')
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
    
def rename(df, app):  # rename columns from merging protocol
    return df.rename({'Pos_'+app: 'Pos','Salary_'+app: 'Salary',
    'Batting Order (Confirmed)_'+app: 'Batting Order (Confirmed)',
    'Proj FP_'+app: 'Proj FP','Actual FP_'+app: 'Actual FP'})
    
def frame_maker(date, numg): # format data frames, cut away fat
    #need to process DK and FD frames separately at first
    pdf1 = pd.read_csv('data/DFN MLB Pitchers DK {}.csv'.format(date))
    pdf1 = pdf1[['Player Name','Pos','Salary','Team','Opp','Proj FP','Actual FP']]
    pdf2 = pd.read_csv('data/DFN MLB Pitchers FD {}.csv'.format(date))
    pdf2 = pdf2[['Player Name','Pos','Salary','Team','Opp','Proj FP','Actual FP']]
    hdf1 = pd.read_csv('data/DFN MLB Hitters DK {}.csv'.format(date))
    hdf1 = hdf1[['Player Name','Pos','Salary','Team','Opp','Batting Order (Confirmed)','Proj FP','Actual FP']]
    hdf1 = hdf1[hdf1['Batting Order (Confirmed)'] != 'x']    #only confirmed ordered batters
    hdf2 = pd.read_csv('data/DFN MLB Hitters FD {}.csv'.format(date))
    hdf2 = hdf2[['Player Name','Pos','Salary','Team','Opp','Batting Order (Confirmed)','Proj FP','Actual FP']]
    hdf2 = hdf2[hdf2['Batting Order (Confirmed)'] != 'x']    #only confirmed ordered batters

    pdf1['Name_Team'] = pdf1['Player Name'] + '_' + pdf1['Team']
    pdf2['Name_Team'] = pdf2['Player Name'] + '_' + pdf1['Team']
    hdf1['Name_Team'] = hdf1['Player Name'] + '_' + hdf1['Team']
    hdf2['Name_Team'] = hdf2['Player Name'] + '_' + hdf2['Team']
    
    pdf = pd.merge(pdf1, pdf2, how = 'inner', on = 'Name_Team')  # fasten FD/DK together
    hdf = pd.merge(hdf1, hdf2, how = 'inner', on = 'Name_Team')

    df = pd.concat([pdf, hdf])
    print(df.columns)
    for colname in ['Player Name', 'Team', 'Opp']: # get rid of columnar dupes
        if not df[colname+'_x'].equals(df[colname+'_y']): 
            print(colname+' not equal.') 
            sys.exit()
        print(colname+'_y')
        df = df.drop(colname+'_y', axis=1).rename({colname+'_x': colname})
    print(df.columns)
    df['Pos_x'] = df['Pos_x'].str.split('/').str[0]#for multi-pos players choose 1st pos
    df['Pos_y'] = df['Pos_y'].str.split('/').str[0]#for multi-pos players choose 1st pos
    df = df[df['Proj FP_x']>0]       # 0 is a bad fantasy score to use
    df = df[df['Proj FP_y']>0]       # 0 is a bad fantasy score to use
    
    teamlist = team_sampler(df, int(numg))  
    df = df[df['Team'].isin(teamlist)]
    
    return df, teamlist

def mask_maker(df, teamlist):
    tothit = 9                          # nine hitters in order
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
    
    st = {3: [], 4: [] , 5: []}  # stacking masks
    for n in [3,4,5]:  # n is consecutive number of batters
        for k in range(tothit):   #  for all 9 possible masks
            stemp = np.array([0]*len(df['Batting Order (Confirmed)']))
            for i in range(n):
                if k+i > tothit: stemp+=(df['Batting Order (Confirmed)']==str(k+i-tothit+1))
                if k+i <= tothit: stemp+=(df['Batting Order (Confirmed)']==str(k+i+1))
            st[n].append(stemp)
    
    #masks in dict for easier transport 
    m = {'p': p,'h': h,'c': c,'b1': b1,'b2':b2,'b3':b3,'ss':ss,'of':of,'tb':tb,'st': st}     
    return m
    
def solver(df, m, params, limits, rosters, platform):
    #need to adjust constraints for FD
    # adjust graming for FD (make sure they all have same indices)
    #order constraint not working sometimes
    #SUCCESSive order constraints for when things run out
    # convert dictionary to actual variable names
    p, h, teambool, st = m['p'], m['h'], m['tb'], m['st']
    c, b1, b2, b3, ss, of = m['c'], m['b1'] , m['b2'], m['b3'], m['ss'], m['of']
    B, stacks, overlap = params['B'], params['stacks'], params['overlap']
    nptch, nhit, no_c, no_1b, no_c1b, no_2b = limits['no_ptch'], limits['no_hit'], limits['no_c'], limits['no_1b'], limits['no_c1b'], limits['no_2b']
    no_3b, no_ss, no_of = limits['no_3b'], limits['no_ss'], limits['no_of']   
    # = no_c + no_1b + no_2b + no_3b + no_ss + no_of
     
    pr = np.array(df['Proj FP'])

    cost = np.array(df['Salary'].to_list())  #cost vector
    d = np.array([float(cost[i]) for i in range(len(cost))])

    v1 = np.array(len(df)*[1])  # vector of ones used for summations

    m = gp.Model()
    m.setParam("LogToConsole",0)
    x = m.addMVar(len(df), vtype=GRB.BINARY) # roster 
    v = m.addMVar(len(teambool)*len(st[0]),vtype=GRB.BINARY)#ensures consecutive hitters
    w = m.addMVar(len(teambool)*len(st[1]),vtype=GRB.BINARY)#ensures consecutive hitters

    #Setup Constraints
    m.addConstr(d @ x <= B)                                  #budget constraint

    m.addConstr(np.diag(p) @ x @ v1 == nptch)           #  num pitchers
    m.addConstr(np.diag(h) @ x @ v1 == nhit)           #  num hitters

    i = 0  # order counter
    for tbarr in teambool: 
        m.addConstr(stacks[0]*(np.diag(tbarr) @ np.diag(p) @ x @ v1) + np.diag(tbarr) @ np.diag(h) @ x @ v1 <= stacks[0])    # no opposing pitchers / only a max of mst hitters
        for odx in range(len(st)):          # consecutive batters constraint
           m.addConstr(np.diag(tbarr) @ np.diag(h) @ np.diag(st[stacks[0]][odx]) @ x @ v1 >= stacks[0]*v[i])
           m.addConstr(np.diag(tbarr) @ np.diag(h) @ np.diag(st[stacks[1]][odx]) @ x @ v1 >= stacks[1]*w[i])
           i += 1
           
    m.addConstr(sum(v) >= 1)    # ensure at least 1 team w order constraint 1
    m.addConstr(sum(w) >= 1)    # ensure at least 1 team w order constraint 2
    m.addConstr(v @ w == 0)     # make sure order constraints distinct
           
    m.addConstr(np.diag(c) @ x  @ v1 <= no_c)  # positional constraints 
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

    solution = np.array([int(v.X) for v in m.getVars() if abs(v.obj) > 1e-6])
    return solution
 
def main():
    date, no_games, max_stck, overlap = sys.argv[1:]
    rosters = []
    q = 0
    stacks = [[5,2], [4,3], [3,3]]
    params = {'B': float(50000),'stacks': stacks[q],  
              'overlap': int(overlap)}
    limits = {'no_ptch': 2, 'no_hit': 8, 'no_c': 1, 'no_1b': 1, 'no_c1b': 2, 
              'no_2b': 1, 'no_3b': 1, 'no_ss': 1, 'no_of': 3}                      
    no_rosters = {'DK':1, 'FD':1}    # 150 DK rosters, #150 FD rosters
    df, teams = frame_maker(date, no_games)
    
    frame = rename(df, 'x')
    masks = mask_maker(frame, teams, params['maxst'])
    while len(rosters) < no_rosters['DK']:  #get DraftKings rosters
        soln=solver(frame, masks, params, limits, rosters, 'DK')
        if len(soln) == 0: q+=1; params['stacks'] = stacks[q]
        else: rosters.append(soln)
        #print(i)
        #print(frame.loc[soln==1][['Player Name','Pos','Salary','Team','Batting Order (Confirmed)']])
        
    frame = rename(date, 'y')
    params['B'] = float(35000) 
    limits = {'no_ptch': 1, 'no_hit': 8, 'no_c': 2, 'no_1b': 2, 'no_c1b': 2,'no_2b': 2, 'no_3b': 2, 'no_ss': 2, 'no_of': 4} 
    masks = mask_maker(frame, teams, params['maxst'])
    while len(rosters) < no_rosters['FD'] + no_rosters['DK']: #get FanDuel rosters
        soln=solver(frame, masks, params, limits, rosters,'FD')
        if len(soln) == 0: q+=1; params['stacks'] = stacks[q]
        else: rosters.append(soln)
        #print(i)
        #print(frame.loc[soln==1][['Player Name','Pos','Salary','Team','Batting Order (Confirmed)']])
        
    grader(frame, rosters, sys.argv[2:])#,
    
if __name__ == "__main__":
    main()
