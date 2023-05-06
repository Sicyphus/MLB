import os,sys

for ovlp in [4]: 
    for i in range(31):
        date = '4_'+str(i+1)
        print(date)
        try:
            print('python code_for_Github.py {}  9 3 {}'.format(date,ovlp))
            os.system('python code_for_Github.py {}  9 3 {}'.format(date,ovlp))
        except:
            print ("failed for date {} overlap {}".format(date, ovlp))
            
            
            
'''
python code_for_Github.py 4_1  9 3 4
python code_for_Github.py 4_2  9 3 4
python code_for_Github.py 4_3  9 3 4
python code_for_Github.py 4_4  9 3 4
python code_for_Github.py 4_5  9 3 4
python code_for_Github.py 4_6  9 3 4
python code_for_Github.py 4_7  9 3 4
python code_for_Github.py 4_8  9 3 4
python code_for_Github.py 4_9  9 3 4
python code_for_Github.py 4_10  9 3 4
python code_for_Github.py 4_11  9 3 4
python code_for_Github.py 4_12  9 3 4
python code_for_Github.py 4_13  9 3 4
python code_for_Github.py 4_14  9 3 4
python code_for_Github.py 4_15  9 3 4
python code_for_Github.py 4_16  9 3 4
python code_for_Github.py 4_17  9 3 4
python code_for_Github.py 4_18  9 3 4
python code_for_Github.py 4_19  9 3 4
python code_for_Github.py 4_20  9 3 4
python code_for_Github.py 4_21  9 3 4
python code_for_Github.py 4_22  9 3 4
python code_for_Github.py 4_23  9 3 4
python code_for_Github.py 4_24  9 3 4
python code_for_Github.py 4_25  9 3 4
python code_for_Github.py 4_26  9 3 4
python code_for_Github.py 4_27  9 3 4
python code_for_Github.py 4_28  9 3 4
python code_for_Github.py 4_29  9 3 4
python code_for_Github.py 4_30  9 3 4
'''
