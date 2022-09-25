import os
import pathlib
from subprocess import check_call
import sys

TOPDIR = pathlib.Path(__file__).absolute().parents[1]

def build_and_test(config, suffix, intermittent):
    try:
        os.unlink('nvm.bin')
    except FileNotFoundError:
        pass

    my_debug = 1
    config = config.copy()
    # somehow a large samples.bin breaks intermittent
    # execution
    if intermittent:
        try:
            config.remove('--all-samples')
        except ValueError:
            pass
        my_debug = 3
    check_call([sys.executable, TOPDIR / 'dnn-models' / 'transform.py', *config])

    check_call(['cmake', '-S', TOPDIR, '-B', 'build', '-DBUILD_MSP432=OFF', f'-DMY_DEBUG={my_debug}'])
    check_call(['make', '-C', 'build'])

    rounds = 100
    power_cycle = 0.01
    if 'cifar10' in config or 'cifar10-cnn' in config:
        rounds=50
        power_cycle = 0.02
    if '--japari' in config:
        power_cycle += 0.01

    run_cmd = ['./build/intermittent-cnn']
    if intermittent:
        run_cmd = [
            sys.executable, TOPDIR / 'exp' / 'run-intermittently.py',
            '--rounds', str(rounds),
            '--interval', str(power_cycle),
            '--suffix', suffix,
            '--compress',
        ] + run_cmd
    check_call(run_cmd, env={'TMPDIR': '/var/tmp'})

def main():
    # preparation
    suffix = os.environ['LOG_SUFFIX']
    config = os.environ['CONFIG'].split(' ')

    build_and_test(config, suffix, intermittent=False)

    # Test intermittent running
    if '--ideal' not in config:
        build_and_test(config, suffix, intermittent=True)

if __name__ == '__main__':
    main()
