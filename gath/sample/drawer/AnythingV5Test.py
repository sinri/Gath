import time

import yaml
from diffusers import StableDiffusionPipeline

from gath.drawer.GathDrawer import GathDrawer
from gath.drawer.GathMetaDrawer import GathMetaDrawer
from gath.sample.drawer.SampleConstant import SampleConstant

if __name__ == '__main__':
    with open('meta/AnythingV5-0001.yml', 'r', encoding='utf-8') as f:
        draw_meta = yaml.safe_load(f)
    print(draw_meta)

    filename = SampleConstant.output_dir + f"\\AnythingV5-{time.time()}.jpg"
    GathMetaDrawer(draw_meta).draw(filename)
    print(f"saved: {filename}")
