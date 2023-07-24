from random import Random
from typing import Union, List, Optional, Callable, Dict, Any

import torch
from PIL.Image import Image
from diffusers import StableDiffusionControlNetPipeline

from gath.drawer.GathDrawKit import GathDrawKit


class GathControlNetDrawer:
    def __init__(self, pipeline: StableDiffusionControlNetPipeline):
        self.__pipeline = pipeline
        self.__drawn = None
        self.__control_net_image: Optional[Union[Image, torch.FloatTensor]] = None

    def set_scheduler(self, scheduler):
        self.__pipeline.scheduler = scheduler
        return self

    def set_control_net_image(self, control_net_image: Optional[Union[Image, torch.FloatTensor]]):
        self.__control_net_image = control_net_image
        return self

    def get_control_net_image(self):
        return self.__control_net_image

    def load_textual_inversion(self, pretrained_model_name_or_path, **kwargs):
        """
        :param pretrained_model_name_or_path: The embedding or hyper network file (.pt)
        :return:
        """
        GathDrawKit.load_textual_inversion(self.__pipeline, pretrained_model_name_or_path, **kwargs)
        print(f'Loaded Textual Inversion (Embedding): {pretrained_model_name_or_path}, {kwargs}')
        return self

    def load_lora_weights(self, checkpoint_path, multiplier, device, dtype):
        GathDrawKit.load_lora_weights_with_multiplier(self.__pipeline, checkpoint_path, multiplier, device, dtype)
        print(f"LOADED LoRA: {checkpoint_path}, {multiplier}, {device}, {dtype}")
        return self

    def turn_off_nsfw_check(self):
        """
        https://github.com/CompVis/stable-diffusion/issues/239#issuecomment-1241838550
        """
        GathDrawKit.turn_off_nsfw_check(self.__pipeline)
        return self

    def draw(
            self,
            prompt: Union[str, List[str]],
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
            seed: Optional[int] = None, device: Optional[str] = None,
    ):
        kwargs = {'image': self.__control_net_image}

        items = [
            'prompt', 'height', 'width', 'num_inference_steps', 'guidance_scale', 'negative_prompt',
            'num_images_per_prompt', 'eta', 'generator', 'latents', 'prompt_embeds', 'negative_prompt_embeds',
            'output_type', 'return_dict',
            'callback', 'callback_steps',
            'cross_attention_kwargs',
            'controlnet_conditioning_scale',
            'guess_mode'
        ]
        for item in items:
            value = eval(item)
            if value is not None:
                kwargs[item] = value

        return self._draw_with_controlnet(
            seed=seed,
            device=device,
            **kwargs
        )

    def _draw_with_controlnet(self, seed: Optional[int] = None, device: Optional[str] = None, **kwargs):
        # self.__pipeline.enable_model_cpu_offload()

        if not kwargs.__contains__('generator'):
            if seed is None:
                seed = Random().randint(0, 100000000)

            if device is not None:
                kwargs['generator'] = torch.Generator(device=device).manual_seed(seed)
            else:
                kwargs['generator'] = torch.manual_seed(seed)

        if device is not None:
            self.__pipeline.to(device)

        self.__drawn = self.__pipeline(**kwargs)
        return self

    def handle_drawn(self, output_image_file: Optional[str] = None):
        image = self.__drawn.images[0]
        if output_image_file is None:
            image.show()
        else:
            image.save(output_image_file)
