from random import Random
from typing import Optional, Union, List, Callable, Any, Dict

from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers import UniPCMultistepScheduler

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
        self.__canny_image: Optional[Image] = None

        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()
        # pipe.enable_xformers_memory_efficient_attention() # cannot work

    def set_canny(self, canny_image):
        self.__canny_image = canny_image
        return self

    def load_canny_from_source_image(self,
                                     base_image_path: str,
                                     low_threshold=100,
                                     high_threshold=200,
                                     ):
        source_image_np_array = np.array(load_image(base_image_path))
        image = cv2.Canny(source_image_np_array, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return self.set_canny(Image.fromarray(image))

    def get_canny_image(self):
        return self.__canny_image



    def draw(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            guess_mode: bool = False,
            device: Optional[str] = None, seed: Optional[int] = None
    ):
        kwargs = {'image': self.__canny_image}

        keys = [
            'prompt',
            'height',
            'width',
            'num_inference_steps',
            'guidance_scale',
            'negative_prompt',
            'num_images_per_prompt',
            'eta',
            'generator',
            'latents',
            'prompt_embeds',
            'negative_prompt_embeds',
            'output_type',
            'return_dict',
            'callback',
            'callback_steps',
            'cross_attention_kwargs',
            'controlnet_conditioning_scale',
            'guess_mode',
        ]
        for key in keys:
            v = eval(key)
            if v is not None:
                kwargs[key] = v

        self._draw_with_controlnet(seed=seed, device=device, **kwargs)



    def __debug(self, image_path: str):
        source_image = load_image(image_path)
        image = np.array(source_image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        # controlnet_canny_model="lllyasviel/sd-controlnet-canny"
        controlnet_canny_model = "E:\\OneDrive\\Leqee\\ai\\repo\\lllyasviel\\sd-controlnet-canny"
        controlnet = ControlNetModel.from_pretrained(controlnet_canny_model, torch_dtype=torch.float16)

        txt2img_model = "E:\\OneDrive\\Leqee\\ai\\repo\\stable-diffusion-v1-5"

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            txt2img_model, controlnet=controlnet, torch_dtype=torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        # pipe.enable_xformers_memory_efficient_attention()

        prompt = ", best quality, extremely detailed"
        prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
        generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]

        output = pipe(
            prompt,
            canny_image,
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
            num_inference_steps=20,
            generator=generator,
        )

        grid = self.image_grid(output.images, 2, 2)
        grid.show()

    def image_grid(self, imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid


if __name__ == '__main__':
    controlnet_canny_model_path = "E:\\OneDrive\\Leqee\\ai\\repo\\lllyasviel\\sd-controlnet-canny"
    txt2img_model_path = "E:\\OneDrive\\Leqee\\ai\\repo\\stable-diffusion-v1-5"
    # source_image_path = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    source_image_path = 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/56d4c63f-5be6-4ef0-b45b-7fdef85b44ad/width=450/09482-number-seed.jpeg'

    controlnet_canny_model = GathCannyControlNet.load_controlnet(controlnet_canny_model_path)
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
    kit.get_canny_image().show()
    kit.draw(
        prompt='masterpiece,best quality,extremely detailed, a policeman',
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=30,
        guidance_scale=8,
    )
    kit.handle_drawn()
