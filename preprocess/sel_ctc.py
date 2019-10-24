
import sys
import os
import numpy as np
src_dir = sys.argv[1]
tar_dir = sys.argv[2]

def sel_ctc(ppgs_dir, save_dir):
    for ctc in os.listdir(ppgs_dir):
        ppgs_path = os.path.join(ppgs_dir, ctc)
        ppgs = np.load(ppgs_path)
        row = len(ppgs)
        d_list = []
        for i in range(row):
            if ppgs[i][0]>0.99:
                d_list.append(i)
        new_ppgs = np.delete(ppgs, d_list, axis=0)
        save_path = os.path.join(save_dir, ctc)
        np.save(save_path, new_ppgs)




if __name__=='__main__':
    sel_ctc(src_dir,tar_dir)
