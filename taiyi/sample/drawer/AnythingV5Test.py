import time

import yaml
from diffusers import StableDiffusionPipeline

from taiyi.drawer.TaiyiDrawer import TaiyiDrawer
from taiyi.drawer.TaiyiMetaDrawer import TaiyiMetaDrawer
from taiyi.sample.drawer.SampleConstant import SampleConstant

def test_raw():
    anything_v5_model = 'E:\\OneDrive\\Leqee\\ai\\civitai\\AnythingV5_v32.safetensors'
    # d = TaiyiDrawer(
    #     anything_v5_model
    # )
    d = StableDiffusionPipeline.from_ckpt(anything_v5_model, use_safetensors=True)

    # 提示词
    prompt = "masterpiece,(best quality),(1girl) ,solo,(high contrast:1.2),(high saturation:1.2), ((hands on the pocket)),((black and white sdress)),looking at viewer,((purple and blue theme:1.3)),((purple and blue background:1.5)),white hair,blue eyes,((walking:1.3)),full body,black footwear,((the blue water on ground reflecting the starry sky and nebula and galaxy)),((from above:1.2)) "
    # 降噪步数 数字越大，时间越长
    num_inference_steps = 20
    # 遵循提示词的级别 数字越大越接近提示词，但图像质量会下降
    guidance_scale = 7
    # 负面提示词
    negative_prompt = "((very long hair:1.3)),messy,ocean,beach,big breast,nsfw,(((pubic))), ((((pubic_hair)))),sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2girl)), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand"

    drawn = d(
        prompt,
        height=960,
        width=640,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt
    )
    image = drawn.images[0]
    # image.show()
    filename = SampleConstant.output_dir + f"\\AnythingV5-{time.time()}.jpg"
    image.save(filename)
    print(f"saved: {filename}")

if __name__ == '__main__':
    # test_raw()

    with open('meta/AnythingV5-0001.yml', 'r', encoding='utf-8') as f:
        draw_meta = yaml.safe_load(f)
    print(draw_meta)

    filename = SampleConstant.output_dir + f"\\AnythingV5-{time.time()}.jpg"
    TaiyiMetaDrawer(draw_meta).draw(filename)
    print(f"saved: {filename}")


