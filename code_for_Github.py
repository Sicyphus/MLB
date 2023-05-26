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
    print(dframe.columns)
    allteams = set(dframe['Team_x'].to_list())
    if len(allteams)/2 < n:
        print('Not enough games for total teams')
        sys.exit()
    selecteams = random.sample(allteams, n)  # select 2n teams
    opps = []
    for team in selecteams:  # for each team, find opponent and add to list
        opp = dframe[dframe['Team_x'] == team]['Opp_x'].to_list()
        if len(set(opp)) > 1: print('Warning: Opponent team ambiguity')
        else: opps.append(opp[0])
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
    if set(allteams1) != set(allteams2):
        print('Error: Team slate for DK/FD not same:')
        print(set(allteams1))
        print(set(allteams2))
        time.sleep(4)
    return allteams1
    
def printframe(dframe, cols):    
    for i in range(len(dframe)):
        print(dframe.iloc[i][cols])

def name_proc(frm, namelbl, teamlbl):
    frm_col = frm[namelbl].str.lower() + '_' + frm[teamlbl]
    bef = [' jr',' sr',r'\s+',r'\.','WSH']
    aft = [r'',r'',r'',r'','WAS']
    for i in range(len(bef)): 
        frm_col.replace(regex=True, inplace=True, to_replace=bef[i], value=aft[i])
    return frm_col

def create_base(nparr):  # create basis vector from positions
    tmplst = []
    for elmt in nparr:
        tmplst.append(2**elmt>>1)
    return tmplst
    
def rename(df, app):  # rename columns from merging protocol
    return df.rename({'Player Name_'+app: 'Player Name','Team_'+app: 'Team', 'Opp_'+app: 'Opp',
    'Pos_'+app: 'Pos','Salary_'+app: 'Salary', 'Batting Order_'+app: 'Batting Order',
    'Proj FP_'+app: 'Proj FP','Actual FP_'+app: 'Actual FP'}, axis=1)
    
def frame_maker(date, numg): # format data frames, cut away fat
    #need to process DK and FD frames separately at first
    pdf1 = pd.read_csv('data/DFN MLB Pitchers DK {}.csv'.format(date))  #pitching
    pdf1 = pdf1[['Player Name','Pos','Salary','Team','Opp','Proj FP','Actual FP']]
    pdf2 = pd.read_csv('data/DFN MLB Pitchers FD {}.csv'.format(date))
    pdf2 = pdf2[['Player Name','Pos','Salary','Team','Opp','Proj FP','Actual FP']]

    hdf1 = pd.read_csv('data/DFN MLB Hitters DK {}.csv'.format(date)) #hitting (DK)
    hdf1 = hdf1[['Player Name','Pos','Salary','Team','Opp','Batting Order (Projected)','Batting Order (Confirmed)','Proj FP','Actual FP']]#combine proj ord w/ confirmed
    maskdkp = hdf1['Batting Order (Projected)'].isin([str(i+1) for i in range(9)])
    hdf1.loc[~maskdkp, 'Batting Order (Projected)'] = '0' 
    maskdkc = hdf1['Batting Order (Confirmed)'].isin([str(i+1) for i in range(9)])
    hdf1.loc[~maskdkc, 'Batting Order (Confirmed)'] = '0' 
    hdf1['Batting Order'] = hdf1[['Batting Order (Projected)','Batting Order (Confirmed)']].max(axis=1)    
    hdf1.drop(['Batting Order (Confirmed)', 'Batting Order (Projected)'], axis=1,inplace=True)
   
    hdf2 = pd.read_csv('data/DFN MLB Hitters FD {}.csv'.format(date)) #hitting (FD)
    hdf2 = hdf2[['Player Name','Pos','Salary','Team','Opp','Batting Order (Projected)','Batting Order (Confirmed)','Proj FP','Actual FP']]#combine proj ord w/ confirmed
    maskfdp = hdf2['Batting Order (Projected)'].isin([str(i+1) for i in range(9)])
    hdf2.loc[~maskfdp, 'Batting Order (Projected)'] = '0' 
    maskfdc = hdf2['Batting Order (Confirmed)'].isin([str(i+1) for i in range(9)])
    hdf2.loc[~maskfdc, 'Batting Order (Confirmed)'] = '0' 
    hdf2['Batting Order'] = hdf2[['Batting Order (Projected)','Batting Order (Confirmed)']].max(axis=1)    
    hdf2.drop(['Batting Order (Confirmed)', 'Batting Order (Projected)'], axis=1,inplace=True)
    
    pdf1['Name_Team'] = name_proc(pdf1,'Player Name','Team')
    pdf2['Name_Team'] = name_proc(pdf2,'Player Name','Team')
    hdf1['Name_Team'] = name_proc(hdf1,'Player Name','Team')
    hdf2['Name_Team'] = name_proc(hdf2,'Player Name','Team')

    pdf = pd.merge(pdf1, pdf2, how = 'inner', on = 'Name_Team')  # fasten FD/DK together
    hdf = pd.merge(hdf1, hdf2, how = 'inner', on = 'Name_Team')

    df = pd.concat([pdf, hdf])

    df['Pos_x'] = df['Pos_x'].str.split('/').str[0]#for multi-pos players choose 1st pos
    df['Pos_y'] = df['Pos_y'].str.split('/').str[0]#for multi-pos players choose 1st pos
    df['Opp_x'].replace(regex=True, inplace=True, to_replace='@', value=r'') # eliminate '@' from opp col
    df['Opp_y'].replace(regex=True, inplace=True, to_replace='@', value=r'') # eliminate '@' from opp col
    df = df[df['Proj FP_x']>0]       # 0 is a bad fantasy score to use
    df = df[df['Proj FP_y']>0]       # 0 is a bad fantasy score to use
    
    if numg == '0': teamlist = team_fromfile(date)  # same day game slate option
    else: teamlist = team_sampler(df, int(numg))  # get random slate
    df = df[df['Team_x'].isin(teamlist)]

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
    ob = []               # opponent masks
    for team in teamlist:     
        tb.append(col_to_npbool(df,'Team',team))
        ob.append(col_to_npbool(df,'Opp',team))                   # team opponent
        #print(team)
        #print(tb[-1])
    
    st = {2:[], 3: [], 4: [] , 5: []}  # stacking masks
    for n in [2, 3, 4, 5]:  # n is consecutive number of batters
        for k in range(1, tothit+1):   #  for all 9 possible masks
            stemp = np.array([0]*len(df['Batting Order']))
            for i in range(n):
                if k+i > tothit: stemp+=(df['Batting Order']==str(k+i-tothit))
                if k+i <= tothit: stemp+=(df['Batting Order']==str(k+i))
            st[n].append(stemp)
    
    #masks in dict for easier transport 
    m = {'p': p,'h': h,'c': c,'b1': b1,'b2':b2,'b3':b3,'ss':ss,'of':of,'tb':tb,'ob':ob,'st': st}     
    return m

def output(rframe, date, rostnum, platform):
    if rostnum == 1: 
        os.system('rm -rf platform/*Rosters*{}*'.format(date))  # reset file if first rodeo
        open('platform/{}Rosters{}.csv'.format('DK', date), 'w').write('P,P,C,1B,2B,3B,SS,OF,OF,OF\n')  # output titles
        open('platform/{}Rosters{}.csv'.format('FD', date), 'w').write('P,C/1B,2B,3B,SS,OF,OF,OF,UTIL\n') 

    filehdl = open('platform/{}Rosters{}.csv'.format(platform, date), 'a')
    plframe = pd.read_csv('platform/{}Salaries{}.csv'.format(platform, date))  # get todays platform data
    
    names = (platform=='DK')*'Name'+(platform=='FD')*'Nickname'  # make new name/team column for matching
    teams = (platform=='DK')*'TeamAbbrev'+(platform=='FD')*'Team'
    outid = (platform=='DK')*'ID'+(platform=='FD')*'Id'  # name used for outputfile
    plframe['Name_Team'] = name_proc(plframe, names, teams)

    posdict = {'P':[],'C':[],'1B':[],'2B':[],'3B':[],'SS':[],'OF':[],'UTIL':[]}
    
    for i, row in rframe.iterrows():
        nametm = row['Name_Team']  
        match = plframe[plframe['Name_Team'] == nametm]
        if len(match) == 0:
            print('Warning: {} not found'.format(name))
        pos = row['Pos']
        if pos in ['P','RP','SP']: posdict['P'].append(match[outid].values[0])
        if pos in ['C']: posdict['C'].append(match[outid].values[0])
        if pos in ['1B']: posdict['1B'].append(match[outid].values[0])
        if pos in ['2B']: posdict['2B'].append(match[outid].values[0])
        if pos in ['3B']: posdict['3B'].append(match[outid].values[0])
        if pos in ['SS']: posdict['SS'].append(match[outid].values[0])
        if pos in ['OF']: posdict['OF'].append(match[outid].values[0])
        
    for strng in ['C','1B','2B','3B','SS']: # under any of these conditions, these are FD results
        if len(posdict[strng]) > 1:         # therefore, put extra player in UTIL category
           posdict['UTIL'] = [posdict[strng].pop()]
    if len(posdict['OF']) > 3:
           posdict['UTIL'] = [posdict['OF'].pop()]
    if platform == 'FD' and len(posdict['C'])+len(posdict['1B']) ==2:
           posdict['UTIL'] = [posdict['1B'].pop()]
    
    for key in posdict.keys():        #   order to comply with platform csv standards
        for idnum in posdict[key]:
            filehdl.write(str(idnum)+',')
    filehdl.write('\n')
          
def main():
    #JESSE look over that delete loop above (deletes three columns...)
    # opposing pitchers (not that bad)
    #problematic fanduel constraints
    #fix line 157
    # @symbol in opp
    date, no_games, overlap = sys.argv[1:]
    q = 0
    rosters = []
    stacks = [[5,3],[4,4], [4,3],[3,3],[3,2]]
    params = {'B': float(50000),'stack': stacks[q],  
              'overlap': int(overlap)}
    limits = {'no_ptch': 2, 'no_hit': 8, 'no_c': 1, 'no_1b': 1, 'no_c1b': 2, 
              'no_2b': 1, 'no_3b': 1, 'no_ss': 1, 'no_of': 3}                      
    no_rosters = {'DK':150, 'FD':150}    # 150 DK rosters, #150 FD rosters
    df, teams = frame_maker(date, no_games)

    frame = rename(df, 'x')         # find top DK rosters
    masks = mask_maker(frame, teams)
    while len(rosters) < no_rosters['DK']:  #get DraftKings rosters
        soln=solver.glp(frame, masks, params, limits, rosters, 'DK')  # alternative: gurobi   
        if len(soln) == 0: q+=1; params['stack'] = stacks[q]; continue
        else: rosters.append(soln)
        print([len(rosters)]+stacks[q])
        print(frame.loc[soln==1][['Player Name','Pos','Salary','Team','Opp','Batting Order']])
        output(frame.loc[soln==1][['Name_Team','Pos']], date, len(rosters), 'DK')
        
    frame = rename(df, 'y')        # find top FD rosters
    params['B'] = float(35000) 
    limits = {'no_ptch': 1, 'no_hit': 8, 'no_c': 2, 'no_1b': 2, 'no_c1b': 2,'no_2b': 2, 'no_3b': 2, 'no_ss': 2, 'no_of': 4} 
    masks = mask_maker(frame, teams)
    q = max(2, q)    # 5/3 & 4/4 stack invalid for FD; advance stack index
    params['stack'] = stacks[q]
    while len(rosters) < no_rosters['FD'] + no_rosters['DK']: #get FanDuel rosters
        soln=solver.glp(frame, masks, params, limits, rosters,'FD')   # alternative: gurobi
        if len(soln) == 0: q+=1; params['stack'] = stacks[q]; continue
        else: rosters.append(soln)
        print([len(rosters)]+stacks[q])
        print(frame.loc[soln==1][['Player Name','Pos','Salary','Team','Opp','Batting Order']])
        output(frame.loc[soln==1][['Name_Team','Pos']], date, len(rosters), 'FD')
        
    grader(frame, rosters, sys.argv[1:])#,
    
if __name__ == "__main__":
    main()
