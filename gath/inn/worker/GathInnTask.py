import os

from nehushtan.logger.NehushtanFileLogger import NehushtanFileLogger

from gath import env
from gath.drawer.GathMetaDrawer import GathMetaDrawer


class GathInnTask:
    def __init__(self, row: dict):
        self.__row = row

    def get_application_id(self):
        return self.__row['application_id']

    def execute(self, logger: NehushtanFileLogger):
        if self.__row['status'] != 'APPLIED':
            raise Exception('status is not APPLIED')

        meta = self.__build_meta_dict()

        output_file = f'{env.inn_output_folder}{os.sep}{self.__row["application_id"]}.jpg'
        drawer = GathMetaDrawer(meta)
        drawer.draw(output_file)

        if not os.path.isfile(output_file):
            raise Exception('cannot find output file')

        # send to OSS?
        # First, put them into OneDrive, that is enough

    def __build_meta_dict(self) -> dict:
        # shallow copy is enough for
        meta = env.inn_base_meta.copy()

        model_part = env.inn_model_dict.get(self.__row['model'])
        if model_part is None:
            raise Exception('model is not available')
        meta['model'] = model_part

        textual_inversion_part = self.__row.get('textual_inversion')
        if textual_inversion_part is not None:
            meta['textual_inversion'] = env.inn_textual_inversion_dict.get(textual_inversion_part)

        meta['height'] = self.__row.get('height')
        meta['width'] = self.__row.get('width')

        meta['prompt'] = self.__row.get('prompt')
        meta['negative_prompt'] = self.__row.get('negative_prompt')

        meta['long_prompt_arg'] = 1
        if len(meta['prompt']) > 77:
            meta['long_prompt_arg'] = int((len(meta['prompt']) + 76) / 77)

        meta['steps'] = self.__row.get('steps')
        meta['cfg'] = self.__row.get('cfg')
        meta['scheduler'] = self.__row.get('scheduler')

        return meta
