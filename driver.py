import os, sys, numpy as np
from matplotlib import pyplot as plt


file_list = os.listdir('archive/one_stack')

'''
with open() as f:
  v = NP.loadtxt(f, delimiter=",", dtype='float', comments="#", skiprows=1, usecols=None)
v_hist = np.ravel(v)   # 'flatten' v
            n, bins, patches = ax1.hist(v_hist, bins=50, normed=1, facecolor='green')
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
'''
'''

score = {'3': [], '4': [], '5': []}
for fl in file_list:
    if fl.split('.')[-2].split('_')[-2] == '9':
        for key in score.keys():
            if fl.split('.')[-2].split('_')[-1] == key:
                sc = open('archive/one_stack/'+fl).readlines()[0].split(',')
                #sc = np.mean(np.array([float(s) for s in sc])[150:300])  # avg, DK
                #sc = np.mean(np.array([float(s) for s in sc])[0:150])   # avg, FD
                #sc = np.max(np.array([float(s) for s in sc])[150:300])  # max, DK
                sc = np.max(np.array([float(s) for s in sc])[0:150])    # max, FD
                score[key].append(float(sc)) 


kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=20, ec="k")
kwargs['facecolor']='green'; plt.hist(score['3'], **kwargs)      
kwargs['facecolor']='red'; plt.hist(score['4'], **kwargs)      
kwargs['facecolor']='blue';plt.hist(score['5'], **kwargs)      
plt.show()
            
print(np.mean(score['3']))
print(np.mean(score['4']))
print(np.mean(score['5']))
'''
            
'''
python code_for_Github.py 4_1  11 5
python code_for_Github.py 4_2  11 5
python code_for_Github.py 4_3  11 5
python code_for_Github.py 4_4  11 5
python code_for_Github.py 4_5  11 5
python code_for_Github.py 4_6  11 5
python code_for_Github.py 4_7  11 5
python code_for_Github.py 4_8  11 5
python code_for_Github.py 4_9  11 5
python code_for_Github.py 4_10  11 5
python code_for_Github.py 4_11  11 5
python code_for_Github.py 4_12  11 5
python code_for_Github.py 4_13  11 5
python code_for_Github.py 4_14  11 5
python code_for_Github.py 4_15  11 5
python code_for_Github.py 4_16  11 5
python code_for_Github.py 4_17  11 5
python code_for_Github.py 4_18  11 5
python code_for_Github.py 4_19  11 5
python code_for_Github.py 4_20  11 5
python code_for_Github.py 4_21  11 5
python code_for_Github.py 4_22  11 5
python code_for_Github.py 4_23  11 5
python code_for_Github.py 4_24  11 5
python code_for_Github.py 4_25  11 5
python code_for_Github.py 4_26  11 5
python code_for_Github.py 4_27  11 5
python code_for_Github.py 4_28  11 5
python code_for_Github.py 4_29  11 5
python code_for_Github.py 4_30  11 5

python code_for_Github.py 4_1  11 4
python code_for_Github.py 4_2  11 4
python code_for_Github.py 4_3  11 4
python code_for_Github.py 4_4  11 4
python code_for_Github.py 4_5  11 4
python code_for_Github.py 4_6  11 4
python code_for_Github.py 4_7  11 4
python code_for_Github.py 4_8  11 4
python code_for_Github.py 4_9  11 4
python code_for_Github.py 4_10  11 4
python code_for_Github.py 4_11  11 4
python code_for_Github.py 4_12  11 4
python code_for_Github.py 4_13  11 4
python code_for_Github.py 4_14  11 4
python code_for_Github.py 4_15  11 4
python code_for_Github.py 4_16  11 4
python code_for_Github.py 4_17  11 4
python code_for_Github.py 4_18  11 4
python code_for_Github.py 4_19  11 4
python code_for_Github.py 4_20  11 4
python code_for_Github.py 4_21  11 4
python code_for_Github.py 4_22  11 4
python code_for_Github.py 4_23  11 4
python code_for_Github.py 4_24  11 4
python code_for_Github.py 4_25  11 4
python code_for_Github.py 4_26  11 4
python code_for_Github.py 4_27  11 4
python code_for_Github.py 4_28  11 4
python code_for_Github.py 4_29  11 4
python code_for_Github.py 4_30  11 4

python code_for_Github.py 4_1  11 3
python code_for_Github.py 4_2  11 3
python code_for_Github.py 4_3  11 3
python code_for_Github.py 4_4  11 3
python code_for_Github.py 4_5  11 3
python code_for_Github.py 4_6  11 3
python code_for_Github.py 4_7  11 3
python code_for_Github.py 4_8  11 3
python code_for_Github.py 4_9  11 3
python code_for_Github.py 4_10  11 3
python code_for_Github.py 4_11  11 3
python code_for_Github.py 4_12  11 3
python code_for_Github.py 4_13  11 3
python code_for_Github.py 4_14  11 3
python code_for_Github.py 4_15  11 3
python code_for_Github.py 4_16  11 3
python code_for_Github.py 4_17  11 3
python code_for_Github.py 4_18  11 3
python code_for_Github.py 4_19  11 3
python code_for_Github.py 4_20  11 3
python code_for_Github.py 4_21  11 3
python code_for_Github.py 4_22  11 3
python code_for_Github.py 4_23  11 3
python code_for_Github.py 4_24  11 3
python code_for_Github.py 4_25  11 3
python code_for_Github.py 4_26  11 3
python code_for_Github.py 4_27  11 3
python code_for_Github.py 4_28  11 3
python code_for_Github.py 4_29  11 3
python code_for_Github.py 4_30  11 3

python code_for_Github.py 4_1  9 5
python code_for_Github.py 4_2  9 5
python code_for_Github.py 4_3  9 5
python code_for_Github.py 4_4  9 5
python code_for_Github.py 4_5  9 5
python code_for_Github.py 4_6  9 5
python code_for_Github.py 4_7  9 5
python code_for_Github.py 4_8  9 5
python code_for_Github.py 4_9  9 5
python code_for_Github.py 4_10  9 5
python code_for_Github.py 4_11  9 5
python code_for_Github.py 4_12  9 5
python code_for_Github.py 4_13  9 5
python code_for_Github.py 4_14  9 5
python code_for_Github.py 4_15  9 5
python code_for_Github.py 4_16  9 5
python code_for_Github.py 4_17  9 5
python code_for_Github.py 4_18  9 5
python code_for_Github.py 4_19  9 5
python code_for_Github.py 4_20  9 5
python code_for_Github.py 4_21  9 5
python code_for_Github.py 4_22  9 5
python code_for_Github.py 4_23  9 5
python code_for_Github.py 4_24  9 5
python code_for_Github.py 4_25  9 5
python code_for_Github.py 4_26  9 5
python code_for_Github.py 4_27  9 5
python code_for_Github.py 4_28  9 5
python code_for_Github.py 4_29  9 5
python code_for_Github.py 4_30  9 5

python code_for_Github.py 4_1  9 4
python code_for_Github.py 4_2  9 4
python code_for_Github.py 4_3  9 4
python code_for_Github.py 4_4  9 4
python code_for_Github.py 4_5  9 4
python code_for_Github.py 4_6  9 4
python code_for_Github.py 4_7  9 4
python code_for_Github.py 4_8  9 4
python code_for_Github.py 4_9  9 4
python code_for_Github.py 4_10  9 4
python code_for_Github.py 4_11  9 4
python code_for_Github.py 4_12  9 4
python code_for_Github.py 4_13  9 4
python code_for_Github.py 4_14  9 4
python code_for_Github.py 4_15  9 4
python code_for_Github.py 4_16  9 4
python code_for_Github.py 4_17  9 4
python code_for_Github.py 4_18  9 4
python code_for_Github.py 4_19  9 4
python code_for_Github.py 4_20  9 4
python code_for_Github.py 4_21  9 4
python code_for_Github.py 4_22  9 4
python code_for_Github.py 4_23  9 4
python code_for_Github.py 4_24  9 4
python code_for_Github.py 4_25  9 4
python code_for_Github.py 4_26  9 4
python code_for_Github.py 4_27  9 4
python code_for_Github.py 4_28  9 4
python code_for_Github.py 4_29  9 4
python code_for_Github.py 4_30  9 4

python code_for_Github.py 4_1  9 3
python code_for_Github.py 4_2  9 3
python code_for_Github.py 4_3  9 3
python code_for_Github.py 4_4  9 3
python code_for_Github.py 4_5  9 3
python code_for_Github.py 4_6  9 3
python code_for_Github.py 4_7  9 3
python code_for_Github.py 4_8  9 3
python code_for_Github.py 4_9  9 3
python code_for_Github.py 4_10  9 3
python code_for_Github.py 4_11  9 3
python code_for_Github.py 4_12  9 3
python code_for_Github.py 4_13  9 3
python code_for_Github.py 4_14  9 3
python code_for_Github.py 4_15  9 3
python code_for_Github.py 4_16  9 3
python code_for_Github.py 4_17  9 3
python code_for_Github.py 4_18  9 3
python code_for_Github.py 4_19  9 3
python code_for_Github.py 4_20  9 3
python code_for_Github.py 4_21  9 3
python code_for_Github.py 4_22  9 3
python code_for_Github.py 4_23  9 3
python code_for_Github.py 4_24  9 3
python code_for_Github.py 4_25  9 3
python code_for_Github.py 4_26  9 3
python code_for_Github.py 4_27  9 3
python code_for_Github.py 4_28  9 3
python code_for_Github.py 4_29  9 3
python code_for_Github.py 4_30  9 3


python code_for_Github.py 4_1  7 6
python code_for_Github.py 4_2  7 6
python code_for_Github.py 4_3  7 6
python code_for_Github.py 4_4  7 6
python code_for_Github.py 4_5  7 6
python code_for_Github.py 4_6  7 6
python code_for_Github.py 4_7  7 6
python code_for_Github.py 4_8  7 6
python code_for_Github.py 4_9  7 6
python code_for_Github.py 4_10  7 6
python code_for_Github.py 4_11  7 6
python code_for_Github.py 4_12  7 6
python code_for_Github.py 4_13  7 6
python code_for_Github.py 4_14  7 6
python code_for_Github.py 4_15  7 6
python code_for_Github.py 4_16  7 6
python code_for_Github.py 4_17  7 6
python code_for_Github.py 4_18  7 6
python code_for_Github.py 4_19  7 6
python code_for_Github.py 4_20  7 6
python code_for_Github.py 4_21  7 6
python code_for_Github.py 4_22  7 6
python code_for_Github.py 4_23  7 6
python code_for_Github.py 4_24  7 6
python code_for_Github.py 4_25  7 6
python code_for_Github.py 4_26  7 6
python code_for_Github.py 4_27  7 6
python code_for_Github.py 4_28  7 6
python code_for_Github.py 4_29  7 6
python code_for_Github.py 4_30  7 6

python code_for_Github.py 4_1  7 5
python code_for_Github.py 4_2  7 5
python code_for_Github.py 4_3  7 5
python code_for_Github.py 4_4  7 5
python code_for_Github.py 4_5  7 5
python code_for_Github.py 4_6  7 5
python code_for_Github.py 4_7  7 5
python code_for_Github.py 4_8  7 5
python code_for_Github.py 4_9  7 5
python code_for_Github.py 4_10  7 5
python code_for_Github.py 4_11  7 5
python code_for_Github.py 4_12  7 5
python code_for_Github.py 4_13  7 5
python code_for_Github.py 4_14  7 5
python code_for_Github.py 4_15  7 5
python code_for_Github.py 4_16  7 5
python code_for_Github.py 4_17  7 5
python code_for_Github.py 4_18  7 5
python code_for_Github.py 4_19  7 5
python code_for_Github.py 4_20  7 5
python code_for_Github.py 4_21  7 5
python code_for_Github.py 4_22  7 5
python code_for_Github.py 4_23  7 5
python code_for_Github.py 4_24  7 5
python code_for_Github.py 4_25  7 5
python code_for_Github.py 4_26  7 5
python code_for_Github.py 4_27  7 5
python code_for_Github.py 4_28  7 5
python code_for_Github.py 4_29  7 5
python code_for_Github.py 4_30  7 5

python code_for_Github.py 4_1  7 4
python code_for_Github.py 4_2  7 4
python code_for_Github.py 4_3  7 4
python code_for_Github.py 4_4  7 4
python code_for_Github.py 4_5  7 4
python code_for_Github.py 4_6  7 4
python code_for_Github.py 4_7  7 4
python code_for_Github.py 4_8  7 4
python code_for_Github.py 4_9  7 4
python code_for_Github.py 4_10  7 4
python code_for_Github.py 4_11  7 4
python code_for_Github.py 4_12  7 4
python code_for_Github.py 4_13  7 4
python code_for_Github.py 4_14  7 4
python code_for_Github.py 4_15  7 4
python code_for_Github.py 4_16  7 4
python code_for_Github.py 4_17  7 4
python code_for_Github.py 4_18  7 4
python code_for_Github.py 4_19  7 4
python code_for_Github.py 4_20  7 4
python code_for_Github.py 4_21  7 4
python code_for_Github.py 4_22  7 4
python code_for_Github.py 4_23  7 4
python code_for_Github.py 4_24  7 4
python code_for_Github.py 4_25  7 4
python code_for_Github.py 4_26  7 4
python code_for_Github.py 4_27  7 4
python code_for_Github.py 4_28  7 4
python code_for_Github.py 4_29  7 4
python code_for_Github.py 4_30  7 4
'''


'''
for ovlp in [4]: 
    for i in range(31):
        date = '4_'+str(i+1)
        print(date)
        try:
            print('python code_for_Github.py {}  9 {}'.format(date,ovlp))
            os.system('python code_for_Github.py {}  9 {}'.format(date,ovlp))
        except:
            print ("failed for date {} overlap {}".format(date, ovlp))
'''            
            
