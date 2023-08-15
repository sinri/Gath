import os

from gath.inn.worker.GathInnWorker import GathInnWorker

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if __name__ == '__main__':
    GathInnWorker().start(60 * 5-30)
