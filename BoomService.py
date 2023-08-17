from gath.inn.boom.BoomWorker import BoomWorker

if __name__ == '__main__':
    boom = BoomWorker()

    for i in range(10):
        boom.boom_one()
