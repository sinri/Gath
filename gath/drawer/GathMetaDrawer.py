import random
from typing import Optional

import torch
from diffusers import AutoencoderKL, StableDiffusionLatentUpscalePipeline
from diffusers.models.modeling_utils import load_state_dict
from transformers import CLIPTokenizer

from gath.drawer.GathDrawer import GathDrawer
from gath.env import inn_vae_dict

class GathMetaDrawer:
    def __init__(self, draw_meta: dict):
        self.__draw_meta = draw_meta
        self.__base_model_string: Optional[str] = None

        self.__prompt: str = self.__draw_meta['prompt']
        self.__prompt.replace('\n', ' ')

        # CLIP length limitation truncate workaround
        self.__long_prompt_arg = 1
        if self.__draw_meta.__contains__('long_prompt_arg'):
            self.__long_prompt_arg = int(self.__draw_meta['long_prompt_arg'])

        # whether turn off NSFW check
        self.__turn_off_nsfw_check = False
        if self.__draw_meta.__contains__('turn_off_nsfw_check'):
            # print(f'[{self.__draw_meta["turn_off_nsfw_check"]}] {type(self.__draw_meta["turn_off_nsfw_check"])}')
            self.__turn_off_nsfw_check = self.__draw_meta['turn_off_nsfw_check'] is True or \
                (self.__draw_meta['turn_off_nsfw_check'] == 'true')

        self.__device = None
        if self.__draw_meta.__contains__("device"):
            self.__device = self.__draw_meta['device']

        self.__upscaler_meta = None

    def __generate_tokenizer(self):
        model_meta = self.__draw_meta['model']
        if not model_meta.__contains__('tokenizer'):
            return None

        tokenizer_meta = model_meta['tokenizer']
        if not isinstance(tokenizer_meta, dict):
            return None

        if tokenizer_meta.__contains__('path'):
            if tokenizer_meta.__contains__('max_length'):
                tokenizer = CLIPTokenizer.from_pretrained(tokenizer_meta['path'],
                                                          max_length=tokenizer_meta['max_length'])
            else:
                tokenizer = CLIPTokenizer.from_pretrained(tokenizer_meta['path'])
        else:
            if not tokenizer_meta.__contains__('name'):
                return None
            if tokenizer_meta.__contains__('max_length'):
                tokenizer = CLIPTokenizer.from_pretrained(tokenizer_meta['name'],
                                                          max_length=tokenizer_meta['max_length'])
            else:
                tokenizer = CLIPTokenizer.from_pretrained(tokenizer_meta['name'])

        return tokenizer

    def __generate_drawer(self):
        tokenizer = self.__generate_tokenizer()

        model_meta = self.__draw_meta['model']

        if model_meta.__contains__('upscaler'):
            self.__upscaler_meta = model_meta['upscaler']

        params = {}

        if model_meta.__contains__('type') and model_meta['type'] == 'ckpt':
            # model is a file in safetensors or ckpt
            self.__base_model_string = model_meta['path']
            # drawer = GathDrawer.from_ckpt(self.__base_model_string, **params)
            drawer = GathDrawer.from_single_file(self.__base_model_string, **params)
        else:
            # model is a dir or a model id str
            if tokenizer is not None:
                params['tokenizer'] = tokenizer
            if self.__long_prompt_arg > 1:
                params['custom_pipeline'] = "lpw_stable_diffusion"

            # vae
            vae_code = self.__draw_meta.get('vae')
            if vae_code != '':
                vae_meta = inn_vae_dict.get(vae_code)
            else:
                vae_meta = model_meta.get('vae')

            if vae_meta is not None and isinstance(vae_meta, dict):
                if vae_meta.__contains__('ckpt'):
                    vae_ckpt = vae_meta.get('ckpt')
                    params['vae'] = load_state_dict(vae_ckpt)
                elif vae_meta.__contains__('path'):
                    vae_path = vae_meta.get('path')
                    params['vae'] = AutoencoderKL.from_pretrained(vae_path)

            if model_meta.__contains__('path'):
                params['local_files_only'] = True
                self.__base_model_string = model_meta.get('path')
                drawer = GathDrawer.from_pretrained(self.__base_model_string, **params)
            else:
                self.__base_model_string = model_meta.get('name')
                drawer = GathDrawer.from_pretrained(self.__base_model_string, **params)

        if self.__device is not None:
            drawer.to_device(self.__device)

        if self.__turn_off_nsfw_check:
            drawer.turn_off_nsfw_check()
            # print("nsfw allowed")

        self.__decide_scheduler(drawer)

        if self.__draw_meta.__contains__('lora'):

            if isinstance(self.__draw_meta['lora'], dict):
                # legacy as a single dict
                lora_meta_list = [self.__draw_meta['lora']]
            else:
                # newer: as list of dict
                lora_meta_list = self.__draw_meta['lora']

            for lora_meta in lora_meta_list:
                checkpoint_path = lora_meta['checkpoint_path']
                multiplier = 1.0
                if lora_meta.__contains__('multiplier'):
                    multiplier = float(lora_meta['multiplier'])

                lora_dtype = torch.float32
                if lora_meta.__contains__('dtype'):
                    if lora_meta['dtype'] == 'fp16':
                        lora_dtype = torch.float16
                    elif lora_meta['dtype'] == 'bf16':
                        lora_dtype = torch.bfloat16
                    else:
                        lora_dtype = torch.float32
                drawer.load_lora_weights_with_multiplier(checkpoint_path, multiplier, self.__device, lora_dtype)

        if self.__draw_meta.__contains__('textual_inversion'):
            if isinstance(self.__draw_meta['textual_inversion'], dict):
                textual_inversion_list = [self.__draw_meta['textual_inversion'], ]
            else:
                textual_inversion_list = self.__draw_meta['textual_inversion']

            for textual_inversion in textual_inversion_list:
                if not isinstance(textual_inversion, dict):
                    raise RuntimeError()
                if textual_inversion.__contains__('name'):
                    textual_inversion_v = textual_inversion['name']
                else:
                    textual_inversion_v = textual_inversion['path']
                drawer.load_textual_inversion(textual_inversion_v)

        if self.__draw_meta.__contains__('clip_skip'):
            clip_skip: int = self.__draw_meta.get('clip_skip')
            if clip_skip>0:
                drawer.skip_clip(self.__base_model_string, clip_skip)

        return drawer

    def __decide_scheduler(self, drawer):
        # euler, euler_a, default as pndm
        scheduler = self.__draw_meta['scheduler']
        if scheduler == 'Euler a':
            drawer.change_scheduler_to_euler_ancestral_discrete()
        elif scheduler == 'Euler':
            drawer.change_scheduler_to_euler_discrete()
        elif scheduler == 'DDIM':
            drawer.change_scheduler_to_ddim()
        elif scheduler == 'DDPM':
            drawer.change_scheduler_to_ddpm()
        elif scheduler == 'DPM++ 2M Karras':
            drawer.change_scheduler_to_dpm_solver_multistep_karras()
        elif scheduler == 'DPM++ 2S':
            drawer.change_scheduler_to_dpm_pp_2s()
        elif scheduler == 'PNDM':
            drawer.change_scheduler_to_pndm()
        elif scheduler == 'DPM++ 2M':
            drawer.change_scheduler_to_dpm_solver_multistep()
        elif scheduler == 'LMS':
            drawer.change_scheduler_to_lms_discrete()
        elif scheduler == 'UniPcM':
            drawer.change_scheduler_to_uni_pc_multistep()
        elif scheduler == 'DPM2':
            drawer.change_scheduler_to_kdpm_2_discrete()
        elif scheduler == 'DPM2 a':
            drawer.change_scheduler_to_dpm2a()
        elif scheduler == 'Heun':
            drawer.change_scheduler_to_heun()
        elif scheduler is None or scheduler == '':
            print("USE DEFAULT SCHEDULER")
        else:
            print("UNSUPPORTED SCHEDULER, USE DEFAULT")

    def draw(self, output_image_file: Optional[str] = None):
        drawer = self.__generate_drawer()

        params = {}

        if self.__draw_meta.__contains__('height'):
            params['height'] = self.__draw_meta['height']
        if self.__draw_meta.__contains__('width'):
            params['width'] = self.__draw_meta['width']
        if self.__draw_meta.__contains__('steps'):
            params['num_inference_steps'] = self.__draw_meta['steps']
        if self.__draw_meta.__contains__('cfg'):
            params['guidance_scale'] = self.__draw_meta['cfg']
        if self.__draw_meta.__contains__('negative_prompt'):
            params['negative_prompt'] = self.__draw_meta['negative_prompt'].replace('\n', ' ')

        if self.__long_prompt_arg > 1:
            params['max_embeddings_multiples'] = self.__long_prompt_arg

        if self.__draw_meta.__contains__("generator"):
            generator_meta = self.__draw_meta['generator']
            if isinstance(generator_meta, dict):

                if generator_meta.__contains__("framework"):
                    framework = generator_meta['framework']
                    if framework != 'torch':
                        raise RuntimeError("Only My Torch!")

                if self.__device is not None:
                    generator = torch.Generator(device=self.__device)
                else:
                    generator = torch.Generator()

                seed = None
                if generator_meta.__contains__('seed'):
                    seed = generator_meta['seed']
                if seed is None:
                    # seed = random.randint(1, 9223372036854775807)
                    seed = random.randint(1, 9999999999)

                generator.manual_seed(seed)

                params['generator'] = generator

        if self.__upscaler_meta is not None:
            upscaler_dtype = torch.float16
            if self.__upscaler_meta.__contains__('dtype'):
                if self.__upscaler_meta['dtype'] == 'fp16':
                    upscaler_dtype = torch.float16
                elif self.__upscaler_meta['dtype'] == 'bf16':
                    upscaler_dtype = torch.bfloat16
                else:
                    upscaler_dtype = torch.float32

            print(f'to build upscaler: {self.__upscaler_meta["model"]}, {upscaler_dtype}')

            upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
                self.__upscaler_meta['model'],
                torch_dtype=upscaler_dtype
            )
            if self.__draw_meta.__contains__("device"):
                upscaler.to(self.__draw_meta['device'])
            drawer.set_upscaler(upscaler)

        if self.__draw_meta.__contains__("device"):
            drawer.to_device(self.__draw_meta['device'])

        drawn = drawer.draw(self.__prompt, **params)

        image = drawn.images[0]
        if output_image_file is None:
            image.show()
        else:
            image.save(output_image_file)
