from nehushtan.logger.NehushtanFileLogger import NehushtanFileLogger

from gath.inn.boom.Brain import Brain
from gath.kit.GathDB import GathDB


class BoomWorker:
    def __init__(self):
        self.__logger = NehushtanFileLogger('GathInnWorker')
        self.__db = GathDB()

    def boom_one(self):
        brain = Brain()
        row = brain.boom()
        self.__logger.info('boom by brain', row)
        id = self.__db.register_one_task(row)
        self.__logger.info(f'registered id: {id}')
