import os

import requests
from nehushtan.logger.NehushtanFileLogger import NehushtanFileLogger

from gath import env
from gath.drawer.GathMetaDrawer import GathMetaDrawer
from gath.inn.worker.AliyunOSSKit import AliyunOSSKit


class GathInnTask:
    def __init__(self, row: dict):
        self.__row = row

    def get_application_id(self):
        return self.__row['application_id']

    def execute(self, logger: NehushtanFileLogger):
        if self.__row['status'] != 'APPLIED':
            raise Exception('status is not APPLIED')

        meta = self.__build_meta_dict()
        # logger.info('meta built', meta)

        output_file = f'{env.inn_output_folder}{os.sep}{self.__row["application_id"]}.jpg'
        drawer = GathMetaDrawer(meta)
        drawer.draw(output_file)

        if not os.path.isfile(output_file):
            raise Exception('cannot find output file')

        logger.info(f'drawn and saved to {output_file}')

        with open(output_file, 'rb') as file_to_upload:
            files = {'file':file_to_upload}
            values = env.inn_gibeah_verification

            r = requests.post(env.inn_gibeah_upload_url, files=files, data=values)
            print(f'uploaded: {r.status_code} | {r.text}')

        # remove
        os.remove(output_file)

    def __build_meta_dict(self) -> dict:
        # shallow copy is enough for
        meta = env.inn_base_meta.copy()

        model_part = env.inn_model_dict.get(self.__row['model'])
        if model_part is None:
            raise Exception('model is not available')
        meta['model'] = model_part

        textual_inversion_rows = self.__row.get('textual_inversion_rows')
        if textual_inversion_rows is not None and len(textual_inversion_rows) > 0:
            meta['textual_inversion'] = []
            for textual_inversion_row in textual_inversion_rows:
                meta['textual_inversion'].append(
                    env.inn_textual_inversion_dict.get(textual_inversion_row['textual_inversion'])
                )

        lora_rows = self.__row.get('lora_rows')
        for lora_row in lora_rows:
            lora = lora_row['lora']
            lora_multiplier = lora_row['lora_multiplier']
            if lora is not None:
                meta['lora'] = {
                    'checkpoint_path': env.inn_lora_dict.get(lora).get('path'),
                    'multiplier': lora_multiplier,
                    'dtype': env.inn_lora_dict.get(lora).get('dtype')
                }

        meta['height'] = self.__row.get('height')
        meta['width'] = self.__row.get('width')

        meta['prompt'] = self.__row.get('prompt')
        meta['negative_prompt'] = self.__row.get('negative_prompt')

        meta['long_prompt_arg'] = 1
        if len(meta['prompt']) > 77:
            meta['long_prompt_arg'] = int((len(meta['prompt']) + 76) / 77)

        meta['steps'] = self.__row.get('steps')
        meta['cfg'] = float(self.__row.get('cfg'))
        meta['scheduler'] = self.__row.get('scheduler')
        meta['vae']=self.__row.get('vae')

        meta['clip_skip']=self.__row.get('clip_skip')

        return meta
