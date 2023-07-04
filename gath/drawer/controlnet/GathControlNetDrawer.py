from random import Random
from typing import Union, List, Optional, Callable, Dict, Any

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline

from gath.drawer.GathDrawer import GathDrawer


class GathControlNetDrawer:
    def __init__(self, pipeline: StableDiffusionControlNetPipeline):
        self.__pipeline = pipeline
        self.__drawn = None

    def set_scheduler(self, scheduler):
        self.__pipeline.scheduler = scheduler
        return self

    @staticmethod
    def build_dummy_safety_checker():
        return lambda images, clip_input: (images, False)

    def turn_off_nsfw_check(self):
        """
        https://github.com/CompVis/stable-diffusion/issues/239#issuecomment-1241838550
        """
        self.__pipeline.safety_checker = self.build_dummy_safety_checker()
        return self

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
