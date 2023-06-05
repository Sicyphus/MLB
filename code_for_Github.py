#python code_for_Github.py 4_26 9 3     # April 26, 9 games, max overlap 3
import pandas as pd, numpy as np, sys, random, scipy.sparse as sp,time,solver, os, csv



def col_to_npbool(dframe, col, val):
    vec = np.array((dframe[col] == val).astype(int).to_list())
    return vec

def grader(df, sol_matrix, basefilename):  # score results\output to file
    filename = 'research/out/{}.csv'.format('_'.join(basefilename))
    scorecum = 0
    with open(filename, 'w') as f:
        for sol in sol_matrix :
            score = df['FPTS'].dot(sol)
            scorecum += score
            f.write(str(score)+',')
        f.write(str(scorecum))

def prep_out(dat):   # reformat output roster files if needed (make space for rosters)
    coloffst = {'DK': 11, 'FD': 10}
    outfile = open('out/DK'+dat+'.csv','r')
    datadk =  list(csv.reader(outfile))
    outfile.close()
    outfile = open('out/DK'+dat+'.csv','w')
    if datadk[0][0] == 'Position':
        spacemkr = ['']*coloffst['DK']
        for i in range(len(datadk)):
            outfile.write(','.join(spacemkr+datadk[i])+'\n')
    outfile.close()
    sys.exit()

def team_sampler(dframe, n): # get sample of 2n teams from team list
    allteams = set(dframe['TEAM_x'].to_list())
    if len(allteams)/2 < n:
        print('Not enough games for total teams')
        sys.exit()
    selecteams = random.sample(allteams, n)  # select 2n teams
    opps = []
    for team in selecteams:  # for each team, find opponent and add to list
        opp = dframe[dframe['TEAM_x'] == team]['OPP_x'].to_list()
        if len(set(opp)) > 1: print('Warning: Opponent team ambiguity')
        else: opps.append(opp[0])
    #print(selecteams+opps)
    return selecteams+opps

def name_discr(dflist): #find name/team discrepancies b/t FD and DK
    for i in range(int(len(dflist)/2)):
        print(column_set_diff(dflist[2*i], dflist[2*i+1], 'PLAYERTEAM'))
    time.sleep(5)

def pos_split(strng):
    if 'OF' in strng: return 'OF'
    return strng.split('/')[0]
    
def printframe(dframe, cols):    
    print('DK/FD Disparities')
    for i in range(len(dframe)):
        print(dframe.iloc[i][cols])

def column_set_diff(dkfrm, fdfrm, col): # find diff b/t FD/DK names/teams sets 
    dkset = set(dkfrm[col].to_list())
    fdset = set(fdfrm[col].to_list()) 
    return [fdset-dkset, fdset-dkset]

def create_base(nparr):  # create basis vector from positions
    tmplst = []
    for elmt in nparr:
        tmplst.append(2**elmt>>1)
    return tmplst
    
def rename(df, app):  # rename columns from merging protocol
    return df.rename({'PLAYER_'+app: 'PLAYER','TEAM_'+app: 'TEAM',
                      'PLATFORMID_'+app: 'PLATFORMID','OPP_'+app: 'OPP',
                      'POS_'+app: 'POS','SALARY_'+app: 'SALARY',                       
                      'BO_'+app: 'BO','FPTS_'+app: 'FPTS'}, axis=1)
    
def frame_maker(date, numg): # format data frames, cut away fat
    #need to process DK and FD frames separately at first
    df_dk = pd.read_csv('in/DK{}.csv'.format(date))  #pitching
    df_dk = df_dk[['PLAYER','POS','SALARY','TEAM','OPP','FPTS']]
    df_dk['PLATFORMID'] = proc_platid()
    df_dk['PLAYERTEAM'] = df_dk['PLAYER']+df_dk['TEAM']       
    pdf_dk = df_dk[df_dk['POS']=='SP'] # pitchers
    hdf_dk = df_dk[df_dk['POS']!='SP'] # hitters
    hdf_dk['BO'] = hdf_dk.groupby('TEAM').cumcount()+1  # make new col 'bat order'
    pdf_dk['BO'] = 0
    
    
    df_fd = pd.read_csv('in/FD{}.csv'.format(date))  #pitching
    df_fd = df_fd[['PLAYER','POS','SALARY','TEAM','OPP','FPTS']]
    df_fd['PLAYERTEAM'] = df_fd['PLAYER']+df_fd['TEAM']       
    pdf_fd = df_fd[df_fd['POS']=='P'] # pitchers
    hdf_fd = df_fd[df_fd['POS']!='P'] # hitters
    hdf_fd['BO'] = hdf_fd.groupby('TEAM').cumcount()+1  # make new col 'bat order'
    pdf_fd['BO'] = 0

    name_discr([pdf_dk, pdf_fd, hdf_dk, hdf_fd])    # check for name discrepancies
    pdf = pd.merge(pdf_dk, pdf_fd, how = 'inner', on = 'PLAYERTEAM') # fasten FD/
    hdf = pd.merge(hdf_dk, hdf_fd, how = 'inner', on = 'PLAYERTEAM') # DK together
    
    df = pd.concat([pdf, hdf])

    # no rows with NaN ;  id must be integers
    for col in ['POS_x','SALARY_x','FPTS_x','POS_y','SALARY_y','FPTS_y']: 
        df = df[df[col].notna()]
    for col in ['PLATFORMID_x','PLATFORMID_y','BO_x','BO_y']:
        df[col] = df[col].astype(int).astype(str) # orders must be inte

    # under construction: make new method that picks OF by default
    df['POS_x'] = df['POS_x'].apply(pos_split)#for multi-pos players choose 1st pos
    df['POS_y'] = df['POS_y'].apply(pos_split)#for multi-pos players choose 1st pos
    
    if numg == '0': teamlist = set(df['TEAM_x'].to_list())#same day game slate opt
    else: 
        teamlist = team_sampler(df, int(numg))  # get random slate
        df = df[df['TEAM_x'].isin(teamlist)]

    return df, teamlist

def mask_maker(df, teamlist):           # create masks to be used for opt. constraints
    tothit = 9                          # nine hitters in order

    sp_ = col_to_npbool(df,'POS','SP')   # pitcher mask (DK)
    p_ = col_to_npbool(df,'POS','P')   # pitcher mask (FD)

    p = np.array([sp_[i]+p_[i] for i in range(len(p_))])   # pitcher mask
    h = np.array([1-p[i] for i in range(len(p))])  # hitter mask

    c = col_to_npbool(df,'POS','C')
    b1 = col_to_npbool(df,'POS','1B')   # infielder mask
    b2 = col_to_npbool(df,'POS','2B')   
    b3 = col_to_npbool(df,'POS','3B')   
    ss = col_to_npbool(df,'POS','SS')   

    of = col_to_npbool(df,'POS','OF')   # outfielder mask

    tb = []               # team masks
    ob = []               # opponent masks
    for team in teamlist:     
        tb.append(col_to_npbool(df,'TEAM',team))
        ob.append(col_to_npbool(df,'OPP',team))                   # team opponent
        #print(team)
        #print(tb[-1])
    
    st = {2:[], 3: [], 4: [] , 5: []}  # stacking masks
    for n in [2, 3, 4, 5]:  # n is consecutive number of batters
        for k in range(1, tothit+1):   #  for all 9 possible masks
            stemp = np.array([0]*len(df['BO']))
            for i in range(n):
                if k+i > tothit: stemp+=(df['BO']==str(k+i-tothit))
                if k+i <= tothit: stemp+=(df['BO']==str(k+i))
            st[n].append(stemp)
    
    #masks in dict for easier transport 
    m = {'p': p,'h': h,'c': c,'b1': b1,'b2':b2,'b3':b3,'ss':ss,'of':of,'tb':tb,'ob':ob,'st': st}     
    return m

def output(rframe, date, rostnum, ng, platform):
    if ng != '0': return     # for random slate, this method not used
    if rostnum == 1: # open output file 
        os.system('rm -rf out/*Rosters*{}*'.format(date)) # reset if first rodeo
        open('out/{}Rosters{}.csv'.format('DK', date), 
             'w').write('P,P,C,1B,2B,3B,SS,OF,OF,OF\n')  # output titles
        open('out/{}Rosters{}.csv'.format('FD', date),          
             'w').write('P,C/1B,2B,3B,SS,OF,OF,OF,UTIL\n') 
    filehdl = open('out/{}Rosters{}.csv'.format(platform, date), 'a')

    posdict = {'SP':[], 'P':[],'C':[],'1B':[],'2B':[],'3B':[],'SS':[],'OF':[],'UTIL':[]}
    
    for i, row in rframe.iterrows():
        posdict[row['POS']].append(row['PLATFORMID'])
        
    # these conds indicate FD results, therefore put extra player in UTIL category 
    for strng in ['C','1B','2B','3B','SS']: 
        if len(posdict[strng]) > 1: 
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
    date, no_games, overlap = sys.argv[1:]
    q = 0
    rosters = []
    stacks = [[5,3],[4,4], [4,3],[3,3],[3,2]]
    params = {'B': float(50000),'stack': stacks[q],  
              'overlap': int(overlap)}
    limits = {'no_ptch': 2, 'no_hit': 8, 'no_c': 1, 'no_1b': 1, 'no_c1b': 2, 
              'no_2b': 1, 'no_3b': 1, 'no_ss': 1, 'no_of': 3}                      
    no_rosters = {'DK':100, 'FD':100}    # 150 DK rosters, #150 FD rosters
    prep_out(date)         # reformat roster output/template file as needed
    df, teams = frame_maker(date, no_games)

    frame = rename(df, 'x')         # find top DK rosters
    masks = mask_maker(frame, teams)
    while len(rosters) < no_rosters['DK']:  #get DraftKings rosters
        soln=solver.glp(frame, masks, params, limits, rosters, 'DK')  # alternative: gurobi   
        if len(soln) == 0: q+=1; params['stack'] = stacks[q]; continue
        else: rosters.append(soln)
        print([len(rosters)]+stacks[q])
        print(frame.loc[soln==1][['PLAYER','PLATFORMID','POS','SALARY','TEAM','OPP','BO']])
        output(frame.loc[soln==1][['PLAYERTEAM','PLATFORMID','POS']], date, len(rosters),
               no_games, 'DK')
        
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
        print(frame.loc[soln==1][['PLAYER','POS','PLATFORMID','SALARY','TEAM','OPP','BO']])
        output(frame.loc[soln==1][['PLAYERTEAM','PLATFORMID','POS']], date, len(rosters), 
               no_games, 'FD')
        
    grader(frame, rosters, sys.argv[1:])#,
    
if __name__ == "__main__":
    main()
