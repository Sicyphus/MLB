#python code_for_Github.py  4_26 9 4 3 # gets data for 4_26, 9 games, at most 4-stack, budget 50K
import pandas as pd, numpy as np, sys, random, scipy.sparse as sp, time, solver, os



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

def team_sampler(dframe, n): # get sample of 2n teams from team list
    allteams = set(dframe['Team'].to_list())
    if len(allteams)/2 < n:
        print('Not enough games for total teams')
        sys.exit()
    selecteams = random.sample(allteams, n)  # select 2n teams
    opps = []
    for team in selecteams:  # for each team, find opponent and add to list
        opp = dframe[dframe['Team'] == team]['Opp'].to_list()
        if len(set(opp)) > 1: print('Warning: Opponent team ambiguity')
        else: opps.append(opp[0].replace('@',''))
    print(selecteams+opps)
    return selecteams+opps

def team_fromfile(date):
    df1 = pd.read_csv('platform/DKSalaries{}.csv'.format(date))
    df2 = pd.read_csv('platform/FDSalaries{}.csv'.format(date))
    allteams1 = list(set(df1['TeamAbbrev'].to_list()))
    allteams2 = list(set(df2['Team'].to_list()))
    team_trans = {'WSH':'WAS'}
    for i in range(len(allteams1)): 
        if allteams1[i] in team_trans.keys(): allteams1[i] = team_trans[allteams1[i]]
    for i in range(len(allteams2)): 
        if allteams2[i] in team_trans.keys(): allteams2[i] = team_trans[allteams2[i]]
    if set(allteams1)&set(allteams2) != set(allteams2):
        print('Warning: Team slate for DK/FD not same:')
        print((set(allteams1)|set(allteams2))-(set(allteams1)&set(allteams2)))
        time.sleep(4)
    return allteams1
    
def printframe(dframe, cols):    
    for i in range(len(dframe)):
        print(dframe.iloc[i][cols])

def name_proc(frm):
    frm_col = frm['Player Name'].str.lower().str.replace(' jr',' ').str.replace(' jr',' ').str.replace(' ','')
    frm_col = frm_col.str.replace('.','') + '_' + frm['Team']
    return frm_col

def create_base(nparr):  # create basis vector from positions
    tmplst = []
    for elmt in nparr:
        tmplst.append(2**elmt>>1)
    return tmplst
    
def rename(df, app):  # rename columns from merging protocol
    return df.rename({'Pos_'+app: 'Pos','Salary_'+app: 'Salary',
    'Batting Order (Projected)_'+app: 'Batting Order (Projected)',
    'Proj FP_'+app: 'Proj FP','Actual FP_'+app: 'Actual FP'}, axis=1)
    
def frame_maker(date, numg): # format data frames, cut away fat
    #need to process DK and FD frames separately at first
    pdf1 = pd.read_csv('data/DFN MLB Pitchers DK {}.csv'.format(date))
    pdf1 = pdf1[['Player Name','Pos','Salary','Team','Opp','Proj FP','Actual FP']]
    pdf2 = pd.read_csv('data/DFN MLB Pitchers FD {}.csv'.format(date))
    pdf2 = pdf2[['Player Name','Pos','Salary','Team','Opp','Proj FP','Actual FP']]
    hdf1 = pd.read_csv('data/DFN MLB Hitters DK {}.csv'.format(date))
    hdf1 = hdf1[['Player Name','Pos','Salary','Team','Opp','Batting Order (Projected)','Proj FP','Actual FP']]
    hdf1 = hdf1[hdf1['Batting Order (Projected)'] != 'x']    #only confirmed ordered batters
    hdf2 = pd.read_csv('data/DFN MLB Hitters FD {}.csv'.format(date))
    hdf2 = hdf2[['Player Name','Pos','Salary','Team','Opp','Batting Order (Projected)','Proj FP','Actual FP']]
    hdf2 = hdf2[hdf2['Batting Order (Projected)'] != 'x']    #only confirmed ordered batters
    
    pdf1['Name_Team'] = pdf1['Player Name'].str.lower().str.replace(' jr',' ').str.replace(' jr',' ').str.replace(' ','').str.replace('.','') + '_' + pdf1['Team']
    pdf2['Name_Team'] = pdf2['Player Name'].str.lower().str.replace(' jr',' ').str.replace(' jr',' ').str.replace(' ','').str.replace('.','') + '_' + pdf2['Team']
    hdf1['Name_Team'] = hdf1['Player Name'].str.lower().str.replace(' jr',' ').str.replace(' jr',' ').str.replace(' ','').str.replace('.','') + '_' + hdf1['Team']
    hdf2['Name_Team'] = hdf2['Player Name'].str.lower().str.replace(' jr',' ').str.replace(' jr',' ').str.replace(' ','').str.replace('.','') + '_' + hdf2['Team']
    
    pdf = pd.merge(pdf1, pdf2, how = 'inner', on = 'Name_Team')  # fasten FD/DK together
    hdf = pd.merge(hdf1, hdf2, how = 'inner', on = 'Name_Team')

    df = pd.concat([pdf, hdf])

    for colname in ['Player Name', 'Team', 'Opp']: # get rid of columnar dupes
        #b=set(df[colname+'_x'].to_list())
        #a=set(df[colname+'_y'].to_list())
        #print((a|b)-(a&b))
        #print(len(a&b))
        #print(len(a|b))
        #if not df[colname+'_x'].equals(df[colname+'_y']): 
        #    print(colname+' not equal.') 
        #    sys.exit()
        df = df.drop(colname+'_y', axis=1).rename({colname+'_x': colname},axis=1)

    df['Pos_x'] = df['Pos_x'].str.split('/').str[0]#for multi-pos players choose 1st pos
    df['Pos_y'] = df['Pos_y'].str.split('/').str[0]#for multi-pos players choose 1st pos
    df = df[df['Proj FP_x']>0]       # 0 is a bad fantasy score to use
    df = df[df['Proj FP_y']>0]       # 0 is a bad fantasy score to use
    
    if numg == '0': teamlist = team_fromfile(date)  # same day game slate option
    else: teamlist = team_sampler(df, int(numg))  # get random slate  
    df = df[df['Team'].isin(teamlist)]

    return df, teamlist

def mask_maker(df, teamlist):
    tothit = 9                          # nine hitters in order
    sp_ = col_to_npbool(df,'Pos','SP')   # pitcher mask
    rp_ = col_to_npbool(df,'Pos','RP')
    p_ = col_to_npbool(df,'Pos','P')    
    p = np.array([sp_[i]+rp_[i]+p_[i] for i in range(len(sp_))])

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
    
    st = {2:[], 3: [], 4: [] , 5: []}  # stacking masks
    for n in [2, 3, 4, 5]:  # n is consecutive number of batters
        for k in range(1, tothit+1):   #  for all 9 possible masks
            stemp = np.array([0]*len(df['Batting Order (Projected)']))
            for i in range(n):
                if k+i > tothit: stemp+=(df['Batting Order (Projected)']==str(k+i-tothit))
                if k+i <= tothit: stemp+=(df['Batting Order (Projected)']==str(k+i))
            st[n].append(stemp)
    
    #masks in dict for easier transport 
    m = {'p': p,'h': h,'c': c,'b1': b1,'b2':b2,'b3':b3,'ss':ss,'of':of,'tb':tb,'st': st}     
    return m

def output(name_col, date, rostnum, platform):
    if rostnum == 1: 
        os.system('rm -rf platform/*Rosters*{}*'.format(date))  # reset file if first rodeo
        open('{}Rosters{}.csv'.format('DK', date), 'a').write('P,P,C,1B,2B,3B,SS,OF,OF,OF')
        open('{}Rosters{}.csv'.format('FD', date), 'a').write('P,C/1B,2B,3B,SS,OF,OF,OF,UTIL') 

    plframe = pd.read_csv('platform/{}Salaries{}.csv'.format(platform, date))
    names = (platform=='DK')*'Name'+(platform=='FD')*'Nickname'
    teams = (platform=='DK')*'TeamAbbrev'+(platform=='FD')*'Team'
    plframe['Name_Team'] = plframe[names].str.lower().str.replace(' jr',' ').str.replace(' jr',' ').str.replace(' ','').str.replace('.','') + '_' + plframe[teams]

    for name in name_col.to_list():
        match = plframe[plframe['Name_Team'] == name]
        if len(match) == 0:
            print('Warning: {} not found'.format(name))
            filehdl.write(match)
            sys.exit()
        #else:
            
    
    #filehdl.close()
    
def main():
    date, no_games, overlap = sys.argv[1:]
    q = 0
    rosters = []
    stacks = [[5,3],[4,4], [4,3],[3,3],[3,2]]
    params = {'B': float(50000),'stack': stacks[q],  
              'overlap': int(overlap)}
    limits = {'no_ptch': 2, 'no_hit': 8, 'no_c': 1, 'no_1b': 1, 'no_c1b': 2, 
              'no_2b': 1, 'no_3b': 1, 'no_ss': 1, 'no_of': 3}                      
    no_rosters = {'DK':1, 'FD':1}    # 150 DK rosters, #150 FD rosters
    df, teams = frame_maker(date, no_games)

    frame = rename(df, 'x')         # find top DK rosters
    masks = mask_maker(frame, teams)
    while len(rosters) < no_rosters['DK']:  #get DraftKings rosters
        soln=solver.glp(frame, masks, params, limits, rosters, 'DK')  # alternative: gurobi   
        if len(soln) == 0: q+=1; params['stack'] = stacks[q]; continue
        else: rosters.append(soln)
        print([len(rosters)]+stacks[q])
        print(frame.loc[soln==1][['Player Name','Pos','Salary','Team','Batting Order (Projected)']])
        output(frame.loc[soln==1]['Name_Team'], date, len(rosters), 'DK')
        
    frame = rename(df, 'y')        # find top FD rosters
    params['B'] = float(35000) 
    limits = {'no_ptch': 1, 'no_hit': 8, 'no_c': 2, 'no_1b': 2, 'no_c1b': 2,'no_2b': 2, 'no_3b': 2, 'no_ss': 2, 'no_of': 4} 
    masks = mask_maker(frame, teams)
    while len(rosters) < no_rosters['FD'] + no_rosters['DK']: #get FanDuel rosters
        soln=solver.glp(frame, masks, params, limits, rosters,'FD')   # alternative: gurobi
        if len(soln) == 0: q+=1; params['stack'] = stacks[q]; continue
        else: rosters.append(soln)
        print([len(rosters)]+stacks[q])
        print(frame.loc[soln==1][['Player Name','Pos','Salary','Team','Batting Order (Projected)']])
        output(frame.loc[soln==1]['Name_Team'], date, len(rosters), 'FD')
        
    grader(frame, rosters, sys.argv[1:])#,
    
if __name__ == "__main__":
    main()
