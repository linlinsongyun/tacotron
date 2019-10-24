import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import blizzard, ljspeech
from hparams import hparams




def write_metadata(agrs):
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  spk_list = {'biaobei':1, 'ts':2}
  with open(os.path.join(out_dir, args.save_txt), 'w', encoding='utf-8') as f:
    for ppgs in os,listdir(args.ppgs_dir):
      ppgs_name = ppgs.split('.npy')[0]
      lpc32 = pps_name + '.mel.npy'
      lcp32_path = os.path.join(args.lpc32_dir, lpc32)
      if os.path.isfile(lpc32_path):
        if 'biaobei' in ppgs_name:
          spk_id = 1
          spk = 'biaobei'
        elif 'ts' in ppgs_name:
          spk_id = 2
          spk = 'ts'
          
        f.write('%s|%s|%s|%s|%d\n'%(ppgs, ppgs, lpc32, spk, spk_id))
      else:
        os.system('echo %s>>wrong.txt'%lpc32)
        print('%s is none'%lpc32_path)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
  parser.add_argument('--output', default='training')
  parser.add_argument('--save_txt', default='data_list.txt')
  parser.add_argument('--ppgs_dir')
  parser.add_argument('--lpc32_dir')
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  args = parser.parse_args()
  write_meta(args)


if __name__ == "__main__":
  main()
