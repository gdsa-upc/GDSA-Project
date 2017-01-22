import os,sys


'''
Usage:
python save_for_kaggle.py rankings_dir
'''

def readfile(f):

    with open(f,'r') as f:
        return f.readlines()

rankings_dir = sys.argv[0]
rankings_dir = "/Users/Marta/Desktop/GDSA/save/rankings/"

f = open(os.path.join(rankings_dir,'ranking.csv'),'w')
f.write('Query,RetrievedDocuments\n')

for split in ['val','test']:

    for query in os.listdir(os.path.join(rankings_dir,"SIFT",split)):
        ranking = readfile(os.path.join(rankings_dir,"SIFT",split,query))

        f.write(query.split('.')[0] + ',')
        for r in ranking:
            f.write(r.rstrip() + ' ')
        f.write('\n')

f.close()
