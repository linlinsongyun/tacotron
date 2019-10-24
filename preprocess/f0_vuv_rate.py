import numpy as np
import sys
import os

ppgs_dir = sys.argv[1]
world_dir = sys.argv[2]
save_dir = sys.argv[3]
os.makedirs(save_dir, exist_ok=True)

def get_data(ppgs_dir, world_dir, save_dir):
    for fi in os.listdir(ppgs_dir):
        ppgs_path = os.path.join(ppgs_dir, fi)
        ppgs = np.load(ppgs_path)
        ppgs_name = fi.split('.npy')
        f0_path = os.path.join(world_dir, 'f0', '%s.f0'%ppgs_name)
        vuv_path = os.path.join(world_dir, 'cmpsub', '%s_vuv.npy'%ppgs_name)
        f0 = np.fromfile(f0_path, np.float64)
        vuv = np.load(vuv_path)
        row = nnlen(ppgs)
        # to be deleted row_th, is blank token
        d_list = []
        # saved row_th
        r_list = []
        for i in range(row):
            if ppgs[i][0]>0.99:
                d_list.append(i)
            else: r_list.append(i)
        new_ppgs = np.delete(ppgs, d_list, axis=0)
        new_f0, new_vuv = modify_f0_vuv(f0, vuv, r_list)
        new_vuv = new_vuv.reshape(new_vuv.shape[0], 1)
        print('ppgs', ppgs.shape)
        print('new_f0', new_f0.shape)
        print('new_vuv', new_vuv.shape)
        result = np.concatenate((new_ppgs, new_f0, new_vuv), axis=1)
        print('result', result.shape)
        save_path = os.path.join(save_dir, fi)
        np.save(save_path, result)

def modify_f0_vuv(f0, vuv, d_list):
    length = len(f0)
    new_f0_out = []
    new_vuv_out = []
    for index in d_list:
        start = 3*index -3
        # 2+ 3*index---f0_frame, then scale (-5+frame, 5+frame)
        end = 3*index +7
        if start < 0:
            start = 0
        if start > length:
            start = d_list[-2]*3 -2
            end = length
        elif end>length:
            end = length
        new_f0 = f0[start : end]
        new_vuv = vuv[start : end]
        new_f0_r = []
        for i in range(len(new_f0)):
            if new_f0[i]>0:
                new_f0_r.append(np.log10(new_f0[i]))
        new_f0_r = np.array(new_f0_r)
        if len(new_f0_r)>0:
            mean_f0 = np.mean(new_f0_r)
            std_f0 = np.std(new_f0_r)
        else:
            mean_f0 = 0
            std_f0 = 0
        new_f0_out.append([mean_f0, std_f0])

        voice_num = len(new_vuv.nonzero()[0])
        voice_per = voice_num/10.
        new_vuv_out.append(voice_per)

    return np.array(new_f0_out), np.array(new_vuv_out)

if __name=='__main__':
    get_data(ppgs_dir, world_dir, save_dir)
