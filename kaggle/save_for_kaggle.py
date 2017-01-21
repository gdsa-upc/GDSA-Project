import os,sys
from get_params import get_params

'''
Usage:
python save_for_kaggle.py rankings_dir
'''

def readfile(f):

    with open(f,'r') as f:
        return f.readlines()



f = open(os.path.join(params['root_save'],'ranking.csv'),'w')
f.write('Query,RetrievedDocuments\n')
params = get_params()
for split in ['val','test']:

    for query in os.listdir(os.path.join(params['root_save'],split)):
        ranking = readfile(os.path.join(params['root_save'],split,query))

        f.write(query.split('.')[0] + ',')
        for r in ranking:
            f.write(r.rstrip() + ' ')
        f.write('\n')

f.close()
