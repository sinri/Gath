import os
from collections import defaultdict
from typing import Union, Dict, Optional

import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler, \
    DPMSolverMultistepScheduler, LMSDiscreteScheduler, PNDMScheduler, UniPCMultistepScheduler, KDPM2DiscreteScheduler, \
    StableDiffusionPipeline, DDPMScheduler, DPMSolverSinglestepScheduler, KDPM2AncestralDiscreteScheduler, \
    HeunDiscreteScheduler
from diffusers.loaders import TextualInversionLoaderMixin, LoraLoaderMixin
from safetensors.torch import load_file
from transformers import CLIPTextModel


class GathDrawKit:
    @staticmethod
    def build_dummy_safety_checker():
        return lambda images, clip_input: (images, False)

    @staticmethod
    def turn_off_nsfw_check(pipeline: DiffusionPipeline):
        """
        https://github.com/CompVis/stable-diffusion/issues/239#issuecomment-1241838550
        """
        # pipeline.safety_checker = GathDrawKit.build_dummy_safety_checker()
        pipeline.safety_checker = None

    @staticmethod
    def skip_clip(
            pipeline: StableDiffusionPipeline,
            basemodel: str,
            clip_skip: int,
            #        torch_dtype=torch.float16
    ):
        text_encoder = CLIPTextModel.from_pretrained(
            basemodel,
            subfolder="text_encoder",
            num_hidden_layers=12 - clip_skip,
            # torch_dtype=torch_dtype
        )
        pipeline.text_encoder = text_encoder

    @staticmethod
    def load_lora_weight(
            pipeline: LoraLoaderMixin,
            pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
            **kwargs
    ):
        """

        :param pipeline:
        :param pretrained_model_name_or_path_or_dict:
        :param kwargs: you may need local_files_only=True
        :return:
        """
        pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict, **kwargs)

    @staticmethod
    def load_textual_inversion(
            pipeline: TextualInversionLoaderMixin,
            pretrained_model_name_or_path,
            **kwargs
    ):
        """
        :param pipeline:
        :param pretrained_model_name_or_path: The embedding or hyper network file (.pt)
        :param kwargs you may need local_files_only=True
        :return:
        """
        pipeline.load_textual_inversion(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def load_lora_weights_with_multiplier(pipeline: DiffusionPipeline, checkpoint_path, multiplier, device, dtype):
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        # load LoRA weight from .safetensors
        state_dict = load_file(checkpoint_path, device=device)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(dtype)
            weight_down = elems['lora_down.weight'].to(dtype)
            alpha = elems['alpha']
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += multiplier * alpha * torch.mm(
                    weight_up.squeeze(3).squeeze(2),
                    weight_down.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

        print(f"LOADED LoRA {checkpoint_path}, {multiplier}, {device}")

    @staticmethod
    def change_scheduler_to_euler_discrete(pipeline: DiffusionPipeline):
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_euler_ancestral_discrete(pipeline: DiffusionPipeline):
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_ddim(pipeline: DiffusionPipeline):
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_ddpm(pipeline: DiffusionPipeline):
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_dpm_solver_multistep(pipeline: DiffusionPipeline):
        """
        DPM++ 2M
        :param pipeline:
        :return:
        """
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_dpm_solver_multistep_karras(pipeline: DiffusionPipeline):
        """
        DPM++ 2M Karras
        :param pipeline:
        :return:
        """
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

    @staticmethod
    def change_scheduler_to_dpm_pp_2s(pipeline: DiffusionPipeline):
        """
        DPM++ 2S
        :param pipeline:
        :return:
        """
        pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_lms_discrete(pipeline: DiffusionPipeline):
        pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_pndm(pipeline: DiffusionPipeline):
        pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_uni_pc_multistep(pipeline: DiffusionPipeline):
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_kdpm_2_discrete(pipeline: DiffusionPipeline):
        """
        DPM2
        Scheduler inspired by DPM-Solver-2 and Algorthim 2 from Karras et al. (2022).
        """
        pipeline.scheduler = KDPM2DiscreteScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_dpm2a(pipeline: DiffusionPipeline):
        """
        DPM2 a
        """
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def change_scheduler_to_heun(pipeline: DiffusionPipeline):
        """
        Heun
        """
        pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config)
