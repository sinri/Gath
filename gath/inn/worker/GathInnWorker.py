import time
from typing import Optional

from nehushtan.logger.NehushtanFileLogger import NehushtanFileLogger

from gath.GathDB import GathDB
from gath.inn.worker.GathInnTask import GathInnTask


class GathInnWorker:

    def __init__(self):
        self.__logger = NehushtanFileLogger('GathInnWorker')
        self.__db = GathDB()

    def __get_one_task_to_do(self) -> Optional[GathInnTask]:
        row = self.__db.read_one_task_to_execute()
        if row is None:
            return None
        return GathInnTask(row)

    def start(self):
        while True:
            try:
                task = self.__get_one_task_to_do()
                if task is not None:
                    self.__logger.info(f'Fetched task to do: {task.get_application_id()}')
                    try:
                        self.__db.declare_one_task_start_running(task.get_application_id())
                        task.execute(self.__logger)
                        self.__db.declare_one_task_done(task.get_application_id())
                    except Exception as e1:
                        self.__logger.exception(f'task {task.get_application_id()} error', e1)
                        self.__db.declare_one_task_failed(task.get_application_id(), f'{e1}')
                else:
                    time.sleep(5.0)
            except Exception as e2:
                self.__logger.exception(f'fetch task failed', e2)
                time.sleep(5.0)


if __name__ == '__main__':
    GathInnWorker().start()
