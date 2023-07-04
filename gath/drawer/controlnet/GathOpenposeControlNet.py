from typing import Union, List, Optional, Callable, Dict, Any

import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
from torch import Generator

from gath.drawer.controlnet.GathControlNetDrawer import GathControlNetDrawer


class GathOpenposeControlNet(GathControlNetDrawer):
    @staticmethod
    def load_openpose_detector(model: str) -> OpenposeDetector:
        """
        Load from `lllyasviel/Annotators` or its local repo.
        :param model:
        :return:
        """
        return OpenposeDetector.from_pretrained(model)

    @staticmethod
    def load_controlnet(controlnet_model: str, **kwargs) -> ControlNetModel:
        """
        Load from `lllyasviel/sd-controlnet-openpose` or its local repo.
        :param controlnet_model:
        :param kwargs: torch_dtype=torch.float16
        :return:
        """
        return ControlNetModel.from_pretrained(controlnet_model, **kwargs)

    @staticmethod
    def load_model_with_controlnet(txt2img_model: str, controlnet: ControlNetModel,
                                   **kwargs) -> StableDiffusionControlNetPipeline:
        return StableDiffusionControlNetPipeline.from_pretrained(
            txt2img_model,
            controlnet=controlnet,
            **kwargs
        )

    def __init__(self, controlnet_mixed_model: StableDiffusionControlNetPipeline):
        super().__init__(controlnet_mixed_model)
        self.__pose = None
        # self.__openpose_detector = openpose_detector
        self.__pipeline: StableDiffusionControlNetPipeline = controlnet_mixed_model

    def detect_pose(self, openpose_detector: OpenposeDetector, base_pose_image_path: str):
        base_pose_image = load_image(base_pose_image_path)
        return self.set_pose(openpose_detector(base_pose_image))

    def set_pose(self, pose):
        self.__pose = pose
        return self

    def get_pose(self):
        return self.__pose

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
        kwargs = {'image': self.__pose}

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

    @staticmethod
    def debug():
        face_pretrained_model_or_path = "E:\\OneDrive\\Leqee\\ai\\repo\\lllyasviel\\Annotators"

        openpose_detector: OpenposeDetector = OpenposeDetector.from_pretrained(face_pretrained_model_or_path)
        print(type(openpose_detector))

        base_pose_image_path = 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/56d4c63f-5be6-4ef0-b45b-7fdef85b44ad/width=450/09482-number-seed.jpeg'
        base_pose_image = load_image(base_pose_image_path)

        #     def __call__(self, input_image, detect_resolution=512, image_resolution=512, include_body=True, include_hand=False, include_face=False, hand_and_face=None, output_type="pil", **kwargs):
        poses = openpose_detector(base_pose_image)
        poses.show()

        # controlnet_model_path="fusing/stable-diffusion-v1-5-controlnet-openpose"
        controlnet_model_path = "E:\\OneDrive\\Leqee\\ai\\repo\\lllyasviel\\sd-controlnet-openpose"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path, torch_dtype=torch.float16
        )

        txt2img_model_path = "E:\\OneDrive\\Leqee\\ai\\repo\\stable-diffusion-v1-5"
        # txt2img_model_path = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            txt2img_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        generator = torch.manual_seed(12312421)
        prompt = "super-hero character, best quality, extremely detailed"
        output = pipe(
            prompt,
            poses,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            generator=generator,
            num_inference_steps=20,
        )
        output.images[0].show()


if __name__ == '__main__':
    controlnet_model_path = "E:\\OneDrive\\Leqee\\ai\\repo\\lllyasviel\\sd-controlnet-openpose"
    controlnet = GathOpenposeControlNet.load_controlnet(controlnet_model_path, torch_dtype=torch.float16)

    txt2img_model_path = "E:\\OneDrive\\Leqee\\ai\\repo\\stable-diffusion-v1-5"
    txt2img_model_path ='E:\\OneDrive\\Leqee\\ai\\repo\\BeenYou'
    txt2img_model = GathOpenposeControlNet.load_model_with_controlnet(txt2img_model_path, controlnet,
                                                                      torch_dtype=torch.float16)

    kit = GathOpenposeControlNet(txt2img_model)
    kit.turn_off_nsfw_check()

    openpose_detector_model_path = "E:\\OneDrive\\Leqee\\ai\\repo\\lllyasviel\\Annotators"
    openpose_detector = GathOpenposeControlNet.load_openpose_detector(openpose_detector_model_path)

    base_pose_image = 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/e7cee70a-ef3d-45b0-9e10-381b4de2370a/width=450/01738-1968231230.jpeg'

    kit.detect_pose(openpose_detector, base_pose_image)
    # kit.get_pose().show()

    kit.set_scheduler(UniPCMultistepScheduler.from_config(txt2img_model.scheduler.config))
    kit.draw(
        prompt='masterpiece, 1girl, snowy day, sitting on the bed side, jk uniform',
        negative_prompt='ugly',
        guidance_scale=14,
        num_inference_steps=30,
        device="cuda"
    )
    kit.handle_drawn('E:\\sinri\\TaiyiDrawer\\output\\openpose-test.jpg')
