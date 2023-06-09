import time

import yaml
from tokenizers import Tokenizer
from transformers import CLIPTokenizer

from gath.drawer.GathDrawer import GathDrawer
from gath.drawer.GathMetaDrawer import GathMetaDrawer
from gath.sample.drawer.SampleConstant import SampleConstant

if __name__ == '__main__':
    while True:
        with open('meta/LeXiaoQi-0003.yml', 'r', encoding='utf-8') as f:
            draw_meta = yaml.safe_load(f)
        print(draw_meta)

        filename = SampleConstant.output_dir + f"\\LeXiaoQi-{time.time()}.jpg"
        GathMetaDrawer(draw_meta).draw(filename)
        print(f"saved: {filename}")

        time.sleep(60)