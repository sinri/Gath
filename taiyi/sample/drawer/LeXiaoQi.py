import time

import torch
from safetensors.torch import safe_open

from taiyi.drawer.TaiyiDrawer import TaiyiDrawer
from taiyi.sample.drawer.SampleConstant import SampleConstant

if __name__ == '__main__':
    safetensors_file="E:\\sinri\\sd-scripts\\workspace\\lxq\\trained_model\\openjourney-lxq-1.safetensors"

    # tensors = {}
    # with safe_open(safetensors_file, framework="pt", device=0) as f:
    #     for k in f.keys():
    #         tensors[k] = f.get_tensor(k)
    #         print(tensors[k])

    model=torch.load(safetensors_file)
    model.eval()

    # # 提示词
    # prompt = "lxq robot"
    # # 降噪步数 数字越大，时间越长
    # num_inference_steps = 75
    # # 遵循提示词的级别 数字越大越接近提示词，但图像质量会下降
    # guidance_scale = 10
    # # 负面提示词
    # # negative_prompt = "雨天"
    #
    # drawn = d.draw(
    #     prompt,
    #     height=512,
    #     width=512,
    #     num_inference_steps=num_inference_steps,
    #     guidance_scale=guidance_scale,
    #     # negative_prompt=negative_prompt
    # )
    # image = drawn.images[0]
    # # image.show()
    # filename = SampleConstant.output_dir + f"\\LeXiaoQi-{time.time()}.jpg"
    # image.save(filename)
    # print(f"saved: {filename}")
