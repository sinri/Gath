import time

from gath import env
from gath.drawer.GathDrawer import GathDrawer


def draw_sample_0000():
    d = GathDrawer.from_pretrained(env.sd_v1_5_model)

    # d.change_scheduler_to_dpm_solver_multistep()
    d.change_scheduler_to_euler_ancestral_discrete()

    d.to_device("cuda")

    # 提示词
    prompt = """
    1girl, walking on the water, wearing a dress, spring, sakura blossoms, looking at viewer,
    """
    # 降噪步数 数字越大，时间越长
    num_inference_steps = 30
    # 遵循提示词的级别 数字越大越接近提示词，但图像质量会下降
    guidance_scale = 8
    # 负面提示词
    negative_prompt = """
    nsfw,EasyNegative,nsfw,(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2),bad composition, inaccurate eyes, extra digit,fewer digits,(extra arms:1.2),2girl,lanjiao5,(feet),shoe,text
    """

    drawn = d.draw(
        prompt,
        height=768,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt
    )
    image = drawn.images[0]
    # image.show()
    filename = env.output_dir + f"\\0000-{time.time()}.jpg"
    image.save(filename)
    print(f"saved: {filename}")


if __name__ == '__main__':
    draw_sample_0000()
