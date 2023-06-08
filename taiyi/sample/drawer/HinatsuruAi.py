import time

import yaml
from tokenizers import Tokenizer
from transformers import CLIPTokenizer

from taiyi.drawer.TaiyiDrawer import TaiyiDrawer
from taiyi.drawer.TaiyiMetaDrawer import TaiyiMetaDrawer
from taiyi.sample.drawer.SampleConstant import SampleConstant

if __name__ == '__main__':
    with open('meta/HinatsuruAi-0001.yml', 'r', encoding='utf-8') as f:
        draw_meta = yaml.safe_load(f)
    print(draw_meta)

    filename = SampleConstant.output_dir + f"\\HinatsuruAi-{time.time()}.jpg"
    TaiyiMetaDrawer(draw_meta).draw(filename)
    print(f"saved: {filename}")
