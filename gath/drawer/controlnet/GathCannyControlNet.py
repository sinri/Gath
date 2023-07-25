import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image

from gath.drawer.controlnet.GathControlNetDrawer import GathControlNetDrawer


class GathCannyControlNet(GathControlNetDrawer):

    @staticmethod
    def load_controlnet(controlnet_model: str, **kwargs):
        """

        :param controlnet_model:
        :param kwargs: torch_dtype=torch.float16
        :return:
        """
        return ControlNetModel.from_pretrained(controlnet_model, **kwargs)

    @staticmethod
    def load_model_with_controlnet(txt2img_model: str, controlnet: ControlNetModel, **kwargs):
        """

        :param txt2img_model:
        :param controlnet:
        :param kwargs: torch_dtype=torch.float16
        :return:
        """
        return StableDiffusionControlNetPipeline.from_pretrained(txt2img_model, controlnet=controlnet, **kwargs)

    def __init__(self, controlnet_mixed_model: StableDiffusionControlNetPipeline):
        super().__init__(controlnet_mixed_model)

    def load_canny_from_source_image(self,
                                     base_image_path: str,
                                     low_threshold=100,
                                     high_threshold=200,
                                     ):
        source_image_np_array = np.array(load_image(base_image_path))
        image = cv2.Canny(source_image_np_array, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return self.set_control_net_image(Image.fromarray(image))

if __name__ == '__main__':
    controlnet_canny_model_path = "E:\\sinri\\HuggingFace\\lllyasviel\\sd-controlnet-canny"
    txt2img_model_path = "E:\\OneDrive\\Leqee\\ai\\repo\\stable-diffusion-v1-5"
    # source_image_path = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    source_image_path = 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/56d4c63f-5be6-4ef0-b45b-7fdef85b44ad/width=450/09482-number-seed.jpeg'

    controlnet_canny_model = GathCannyControlNet.load_controlnet(controlnet_canny_model_path,local_files_only=True)
    txt2img_model = GathCannyControlNet.load_model_with_controlnet(txt2img_model_path, controlnet_canny_model)
    kit = GathCannyControlNet(txt2img_model)
    # x.debug(source_image_path)

    kit.set_scheduler(
        UniPCMultistepScheduler.from_config(txt2img_model.scheduler.config)
    )
    kit.load_canny_from_source_image(
        source_image_path,
        low_threshold=150,
        high_threshold=200
    )
    kit.get_control_net_image().show()
    kit.draw(
        prompt='masterpiece,best quality,extremely detailed, a policeman',
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=30,
        guidance_scale=8,
    )
    kit.handle_drawn()
