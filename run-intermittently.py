import argparse
import logging
import os.path
import signal
import tempfile
import sys
from subprocess import Popen, TimeoutExpired

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('run-intermittently')

CHUNK_SIZE = 2000
CHUNK_LINES = 20

def run_one_inference(program, interval, logfile):
    while True:
        with Popen([program, '1'], stdout=logfile, stderr=logfile) as proc:
            try:
                outs, errs = proc.communicate(timeout=interval)
            except TimeoutExpired:
                proc.send_signal(signal.SIGINT)
            proc.wait()
            if proc.returncode in (1, -signal.SIGFPE):
                logger.error('Program crashed!')
                sys.exit(1)
            if proc.returncode == 0:
                logfile.seek(0, 2)
                file_size = logfile.tell()
                logfile.seek(-(CHUNK_SIZE if file_size > CHUNK_SIZE else file_size), 2)
                last_chunk = logfile.read()
                print('\n'.join(last_chunk.decode('ascii').split('\n')[-CHUNK_LINES:]))
                return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=0)
    parser.add_argument('--interval', type=float, default=0.01)
    parser.add_argument('program')
    args = parser.parse_args()

    logfile_path = os.path.join(tempfile.gettempdir(), 'intermittent-cnn')
    rounds = 0
    while True:
        with open(logfile_path, mode='w+b') as logfile:
            run_one_inference(args.program, args.interval, logfile)
            rounds += 1
            if args.rounds and rounds >= args.rounds:
                break


if __name__ == '__main__':
    main()
