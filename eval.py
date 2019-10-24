import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer





def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  for fi in os.listdir(args.ppgs_dir):
    ppgs_path = os.path.join(ppgs_dir, fi)
    ppgs = np.load(ppgs_path)
    speaker_list = {'biaobei':1, ts':2}
    spk = 'biaobei'
    spk_id  =speaker_list[spk]
    lpc = synth.synthesize(spk_id, ppgs)
    out_path = os.path.join(args.out_dir, 'lpc32)
    os.makedisr(out_path, exist_ok=True)
    path = os.path.join(out_path, '%s_%s.npy'%(ppgs_name, spk))
    np.save(path, lpc)
    print('saved', path)
  
  


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--ppgs_dir', help='ppgs_dir')
  parser.add_argument('--out_dir', help='save_dir')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
