import time

from taiyi.drawer.TaiyiDrawer import TaiyiDrawer
from taiyi.sample.drawer.SampleConstant import SampleConstant

if __name__ == '__main__':
    d = TaiyiDrawer(SampleConstant.model)

    # 提示词
    prompt = ["森林", "小路", "蘑菇", "油画"]
    # 降噪步数 数字越大，时间越长
    num_inference_steps = 75
    # 遵循提示词的级别 数字越大越接近提示词，但图像质量会下降
    guidance_scale = 10
    # 负面提示词
    # negative_prompt = ["雨天"]

    drawn = d.draw(
        prompt,
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        # negative_prompt=negative_prompt
    )
    image = drawn.images[0]
    # image.show()
    filename = SampleConstant.output_dir + f"\\MoriNoNaka-{time.time()}.jpg"
    image.save(filename)
    print(f"saved: {filename}")
