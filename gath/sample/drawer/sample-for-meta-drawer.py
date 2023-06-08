import time

import yaml

from gath.drawer.GathMetaDrawer import GathMetaDrawer
from gath.sample import env


def draw_with_any_meta_sample(meta_file_name: str):
    with open('meta/' + meta_file_name, 'r', encoding='utf-8') as f:
        draw_meta = yaml.safe_load(f)
    print(draw_meta)

    filename = env.output_dir + f"\\{meta_file_name}-{time.time()}.jpg"
    GathMetaDrawer(draw_meta).draw(filename)
    print(f"saved: {filename}")


def draw_with_meta_sample_AnythingV5_0001():
    draw_with_any_meta_sample('AnythingV5-0001.yml')


def draw_with_meta_sample_LeXiaoQi_0003():
    draw_with_any_meta_sample('LeXiaoQi-0003.yml')


if __name__ == '__main__':
    draw_with_meta_sample_LeXiaoQi_0003()
