import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image

from gath.drawer.controlnet.GathOpenposeControlNet import GathOpenposeControlNet


def test1(controlnet_model_path: str, txt2img_model_path: str):
    controlnet = GathOpenposeControlNet.load_controlnet(controlnet_model_path, torch_dtype=torch.float16)

    txt2img_model = GathOpenposeControlNet.load_model_with_controlnet(txt2img_model_path, controlnet,
                                                                      torch_dtype=torch.float16)
    kit = GathOpenposeControlNet(txt2img_model)
    kit.turn_off_nsfw_check()

    openpose_img=load_image('E:\\sinri\\TaiyiDrawer\\workspace\\openpose\\pose1.png')
    kit.set_control_net_image(openpose_img)

    kit.set_scheduler(UniPCMultistepScheduler.from_config(txt2img_model.scheduler.config))
    kit.draw(
        prompt='masterpiece, 1girl, snowy day, sitting on the bed side, jk uniform',
        negative_prompt='ugly',
        guidance_scale=14,
        num_inference_steps=30,
        device="cuda"
    )
    kit.handle_drawn('E:\\sinri\\TaiyiDrawer\\output\\openpose-test.jpg')


if __name__ == '__main__':
    controlnet_model_path = "E:\\sinri\\HuggingFace\\lllyasviel\\sd-controlnet-openpose"
    txt2img_model_path = 'E:\\sinri\\HuggingFace\\BeenYou'
    test1(controlnet_model_path, txt2img_model_path)
