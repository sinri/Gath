from typing import Tuple, Optional

from nehushtan.mysql.MySQLAnyTable import MySQLAnyTable
from nehushtan.mysql.MySQLCondition import MySQLCondition
from nehushtan.mysql.MySQLKit import MySQLKit
from nehushtan.mysql.MySQLKitConfig import MySQLKitConfig

from gath import env


class GathDB:
    def __init__(self):
        config = MySQLKitConfig(env.mysql_config)
        self.__db = MySQLKit(config)

    def build_inn_application_table(self):
        return InnApplicationTable(self.__db, 'inn_application')

    def register_one_task(self, row: dict):
        """
        row contains
        # model: str,
        # height: int,
        # width: int,
        # textual_inversion: Optional[str],
        # lora: Optional[str],
        # lora_multiplier: Optional[float],
        # prompt: str,
        # negative_prompt: Optional[str],
        # steps: Optional[int],
        # cfg: float,
        # scheduler: str,
        # seed: Optional[int],
        """
        row['status'] = 'APPLIED'
        row['apply_time'] = MySQLAnyTable.now()
        result = self.build_inn_application_table().insert_one_row(row)
        if not result.is_executed():
            raise Exception(result.get_error())
        return result.get_last_inserted_id()

    def read_one_task(self, application_id) -> dict:
        result = self.build_inn_application_table() \
            .select_in_table() \
            .add_select_field('*') \
            .add_condition([MySQLCondition.make_equal('application_id', application_id)]) \
            .query_for_result_as_tuple_of_dict()
        if not result.is_queried():
            raise Exception(result.get_error())
        return result.get_fetched_first_row_as_dict()

    def read_task_page(self, page=1, page_size=10) -> Tuple[dict]:
        result = self.build_inn_application_table() \
            .select_in_table() \
            .add_select_field('*') \
            .set_limit(page_size) \
            .set_offset((page - 1) * page_size) \
            .query_for_result_as_tuple_of_dict()
        if not result.is_queried():
            raise Exception(result.get_error())
        return result.get_fetched_rows_as_tuple()

    def read_one_task_to_execute(self) -> Optional[dict]:
        result = self.build_inn_application_table() \
            .select_in_table() \
            .add_select_field('*') \
            .add_conditions([MySQLCondition.make_equal('status', 'APPLIED'), ]) \
            .set_sort_expression('application_id') \
            .set_limit(1) \
            .query_for_result_as_tuple_of_dict()
        if not result.is_queried():
            raise Exception(result.get_error())
        rows = result.get_fetched_rows_as_tuple()
        if len(rows)>0:
            return rows[0]
        else:
            return None

    def decalre_one_task_start_running(self, application_id):
        result = self.build_inn_application_table() \
            .update_rows(
            [
                MySQLCondition.make_equal('application_id', application_id),
                MySQLCondition.make_equal('status', 'APPLIED')
            ],
            {
                'status': 'RUNNING',
                'start_time': MySQLAnyTable.now(),
            }
        )
        if not result.is_executed():
            raise Exception(result.get_error())

    def declare_one_task_done(self, application_id):
        result = self.build_inn_application_table() \
            .update_rows(
            [
                MySQLCondition.make_equal('application_id', application_id),
                MySQLCondition.make_equal('status', 'RUNNING')
            ],
            {
                'status': 'DONE',
                'finish_time': MySQLAnyTable.now(),
            }
        )
        if not result.is_executed():
            raise Exception(result.get_error())

    def declare_one_task_failed(self, application_id, feedback: str):
        result = self.build_inn_application_table() \
            .update_rows(
            [
                MySQLCondition.make_equal('application_id', application_id),
                MySQLCondition.make_equal('status', 'RUNNING')
            ],
            {
                'status': 'FAILED',
                'finish_time': MySQLAnyTable.now(),
                'feedback': feedback
            }
        )
        if not result.is_executed():
            raise Exception(result.get_error())


class InnApplicationTable(MySQLAnyTable):
    def __init__(self, mysql_kit: MySQLKit, table_name: str):
        super().__init__(mysql_kit, table_name)
