import random
from typing import Optional

import torch
from transformers import CLIPTokenizer

from gath.drawer.GathDrawer import GathDrawer


class GathMetaDrawer:
    def __init__(self, draw_meta: dict):
        self.__draw_meta = draw_meta

        self.__prompt: str = self.__draw_meta['prompt']
        self.__prompt.replace('\n', ' ')

        # CLIP length limitation truncate workaround
        self.__long_prompt_arg = 1
        if self.__draw_meta.__contains__('long_prompt_arg'):
            self.__long_prompt_arg = self.__draw_meta['long_prompt_arg']

        # whether turn off NSFW check
        self.__turn_off_nsfw_check = False
        if self.__draw_meta.__contains__('turn_off_nsfw_check'):
            self.__turn_off_nsfw_check = self.__draw_meta['turn_off_nsfw_check']

        self.__device = None
        if self.__draw_meta.__contains__("device"):
            self.__device = self.__draw_meta['device']

    def __generate_tokenizer(self):
        if not self.__draw_meta.__contains__('tokenizer'):
            return None

        tokenizer_meta = self.__draw_meta['tokenizer']
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

        params = {}

        if model_meta.__contains__('type') and model_meta['type'] == 'ckpt':
            # model is a file in safetensors or ckpt
            model_path: str = model_meta['path']
            drawer = GathDrawer.from_ckpt(model_path, **params)
        else:
            # model is a dir or a model id str
            if tokenizer is not None:
                params['tokenizer'] = tokenizer
            if self.__long_prompt_arg > 1:
                params['custom_pipeline'] = "lpw_stable_diffusion"

            if model_meta.__contains__('path'):
                drawer = GathDrawer.from_pretrained(model_meta['path'], **params)
            else:
                drawer = GathDrawer.from_pretrained(model_meta['name'], **params)

        if self.__device is not None:
            drawer.to_device(self.__device)

        if self.__turn_off_nsfw_check:
            drawer.turn_off_nsfw_check()

        self.__decide_scheduler(drawer)

        if self.__draw_meta.__contains__('lora'):
            lora_meta = self.__draw_meta['lora']
            checkpoint_path = lora_meta['checkpoint_path']
            multiplier = 1
            if lora_meta.__contains__('multiplier'):
                multiplier = lora_meta['multiplier']
            drawer.load_lora_weights(checkpoint_path, multiplier, self.__device)

        if self.__draw_meta.__contains__('textual_inversion'):
            textual_inversion = self.__draw_meta['textual_inversion']
            if not isinstance(textual_inversion, dict):
                raise RuntimeError()
            if textual_inversion.__contains__('name'):
                textual_inversion_v = textual_inversion['name']
            else:
                textual_inversion_v = textual_inversion['path']
            drawer.load_textual_inversion(textual_inversion_v)

        return drawer

    def __decide_scheduler(self, drawer):
        # euler, euler_a, default as pndm
        scheduler = self.__draw_meta['scheduler']
        if scheduler == 'euler_a':
            drawer.change_scheduler_to_euler_ancestral_discrete()
        elif scheduler == 'euler':
            drawer.change_scheduler_to_euler_discrete()
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
                    seed = random.randint(1, 9223372036854775807)

                generator.manual_seed(seed)

                params['generator'] = generator

        if self.__draw_meta.__contains__("device"):
            drawer.to_device(self.__draw_meta['device'])

        drawn = drawer.draw(self.__prompt, **params)

        image = drawn.images[0]
        if output_image_file is None:
            image.show()
        else:
            image.save(output_image_file)
