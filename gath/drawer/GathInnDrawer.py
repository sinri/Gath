from abc import abstractmethod
from typing import Optional, Union, List, Tuple

import torch
from diffusers import StableDiffusionLatentUpscalePipeline
from transformers import CLIPTokenizer


class GathInnDrawer:
    def __init__(self, draw_meta: dict):
        self.__dmr = DrawMetaReader(draw_meta)

    def draw(self, output_image_file: Optional[str] = None):
        controlnet_type = self.__dmr.read_controlnet_type()

        if controlnet_type == 'canny':
            drawer = CannyDrawer(self.__dmr)
        elif controlnet_type == 'openpose':
            drawer = OpenposeDrawer(self.__dmr)
        else:
            drawer = PureDrawer(self.__dmr)

        drawer.draw(output_image_file)


class DrawMetaReader:
    def __init__(self, draw_meta: dict):
        self.__draw_meta = draw_meta

    def read(self, keychain: Union[str, List[str], Tuple[str]], defaultValue=None):
        if type(keychain) == str:
            return self.__draw_meta.get(keychain, defaultValue)
        else:
            ptr = self.__draw_meta
            for i in range(len(keychain)):
                key = keychain[i]

                if i + 1 == len(keychain):
                    return ptr.get(key, defaultValue)
                else:
                    ptr = ptr.get(key)
                    if type(ptr) != dict:
                        return defaultValue

    def read_local_files_only(self) -> bool:
        return self.read('local_files_only', 'true') == 'true'

    def read_controlnet_type(self) -> Optional[str]:
        return self.read(['controlnet', 'type'])

    def read_prompt(self) -> str:
        return self.read('prompt', '').replace('\n', ' ')

    def read_long_prompt_arg(self) -> int:
        return self.read("long_prompt_arg", 1)

    def read_turn_off_nsfw_check(self) -> bool:
        return self.read("turn_off_nsfw_check", "false") == 'true'

    def read_device(self) -> Optional[str]:
        return self.read('device', None)

    def read_model_tokenizer_path(self) -> Optional[str]:
        return self.read(['model', 'tokenizer', 'path'])

    def read_model_tokenizer_name(self) -> Optional[str]:
        return self.read(['model', 'tokenizer', 'name'])

    def read_model_tokenizer_max_length(self) -> Optional[int]:
        return self.read(['model', 'tokenizer', 'max_length'])

    def generate_tokenizer(self) -> Optional[CLIPTokenizer]:
        tokenizer_code = self.read_model_tokenizer_path()
        if tokenizer_code is None:
            tokenizer_code = self.read_model_tokenizer_name()
        if tokenizer_code is None:
            return None

        tokenizer_max_length = self.read_model_tokenizer_max_length()
        if tokenizer_max_length is None:
            return CLIPTokenizer.from_pretrained(tokenizer_code)
        else:
            return CLIPTokenizer.from_pretrained(tokenizer_code,
                                                 max_length=tokenizer_max_length)

    def read_model_upscaler_model(self) -> Optional[str]:
        return self.read(['model', 'upscaler', 'model'])

    def read_model_upscaler_dtype(self) -> Optional[str]:
        return self.read(['model', 'upscaler', 'dtype'])

    def generate_upscaler(self) -> Optional[StableDiffusionLatentUpscalePipeline]:
        upscaler_model_key = self.read_model_upscaler_model()
        if upscaler_model_key is None:
            return None

        upscaler_dtype_key = self.read_model_upscaler_dtype()
        if upscaler_dtype_key == 'fp32' or upscaler_dtype_key == 'full' or upscaler_dtype_key == 'float':
            upscaler_dtype = torch.float32
        elif upscaler_dtype_key == 'bf16':
            upscaler_dtype = torch.bfloat16
        elif upscaler_dtype_key == 'fp16':
            upscaler_dtype = torch.float16
        else:
            upscaler_dtype = torch.float16

        print(f'to build upscaler: {upscaler_model_key}, {upscaler_dtype}')

        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            upscaler_model_key,
            torch_dtype=upscaler_dtype
        )

        device = self.read_device()
        if device is not None:
            upscaler.to(device)

        return upscaler

    def read_model_type(self) -> str:
        """
        return repo or ckpt.
        """
        return self.read(['model', 'type'], "repo")

    def read_model_path(self) -> str:
        return self.read(['model', 'path'])

    def read_vae_path(self) -> Optional[str]:
        return self.read(['vae', 'path'])


class ExtraLora:
    def __init__(self, item: dict):
        self.__item = item

    def read_checkpoint_path(self) -> str:
        return self.__item.get('checkpoint_path')

    def read_multiplier(self, defaultValue=1) -> int:
        return self.__item.get('multiplier', defaultValue)

    def read_dtype(self) -> Optional[str]:
        return self.__item.get('dtype')


class ExtraTextualInversion:
    def __init__(self, item: dict):
        self.__item = item


class AnyDrawer:
    def __init__(self, dmr: DrawMetaReader):
        self.__dmr = dmr

    @abstractmethod
    def draw(self, output_image_file: Optional[str] = None):
        pass


class PureDrawer(AnyDrawer):
    def __init__(self, dmr: DrawMetaReader):
        super().__init__(dmr)

    def draw(self, output_image_file: Optional[str] = None):
        GathDrawer.from_pretrained()


class CannyDrawer(AnyDrawer):
    def __init__(self, dmr: DrawMetaReader):
        super().__init__(dmr)

    def draw(self, output_image_file: Optional[str] = None):
        pass


class OpenposeDrawer(AnyDrawer):
    def __init__(self, dmr: DrawMetaReader):
        super().__init__(dmr)

    def draw(self, output_image_file: Optional[str] = None):
        pass
