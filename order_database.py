#!/usr/bin/env python


import fnmatch
import os
from shutil import copyfile
import argparse
import numpy as np

def order_files(path, filesdb, rank):

    files=[]
    filename=str("")
    flag=True
    for file in filesdb:
        kk=file.split("_")
        kkk=kk[-1]
        files.append(float(kkk[:-4]))
        if flag:
            flag=False
            for i in range(len(kk)-1):
                filename+=kk[i]+"_"
    files.sort()
    formattedfiles = ["%0.6e" % member for member in files]
    lfiles=formattedfiles
    kfiles=[]

    for i in range(len(lfiles)):

        kfiles.append(path + filename + str(lfiles[i]) + ".csv")

    kkl= ["%05i" % member for member in range(len(kfiles))]

    newnames=[]
    for i in range(len(kfiles)):
        newnames.append(path + "database_"+kkl[i]+".csv")

    for i in range(len(kfiles)):
        if rank==0:
            print "Copy %s to %s" %(kfiles[i],newnames[i])
            copyfile(kfiles[i],newnames[i])
    if rank:
        np.savetxt(path+"times_database.dat", files)
    return (np.array(files, dtype=np.float64, order='C'), newnames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filesdb', nargs='+', help='files')
    args = parser.parse_args()

    (times, newnames)=order_files(args.filesdb)