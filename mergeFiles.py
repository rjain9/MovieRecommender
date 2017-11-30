import sys
from os import listdir
from os.path import isfile, join

mypath = './training_set/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

mergedFile = 'NewMergedFile.txt'
c = 0
# print onlyfiles
for f in onlyfiles:
    c += 1
    if c % 100 == 0:
        print c
        break
    with open(mypath + f) as f:
        lines = f.readlines()
        movieId = lines[0].split(':')[0]
        newlines = [l.strip('\n')+','+movieId+'\n' for l in lines[1:]]
        with open(mergedFile, "a") as f1:
            f1.writelines(newlines)