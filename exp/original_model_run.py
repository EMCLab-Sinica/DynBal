import argparse
import pathlib
import sys

TOPDIR = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(TOPDIR / 'dnn-models'))

from configs import configs
from utils import load_model, run_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=configs.keys())
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--model-variant', type=str, default='')
    parser.add_argument('--save-file')
    args = parser.parse_args()

    if args.limit == 0:
        args.limit = None

    config = configs[args.config]
    model = load_model(config, model_variant=args.model_variant, for_deployment=False)
    model_data = config['data_loader'](train=False)
    run_model(model, model_data, args.limit,
              verbose=not args.save_file, save_file=args.save_file)

if __name__ == '__main__':
    main()
