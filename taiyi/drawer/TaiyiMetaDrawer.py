from typing import Optional

from transformers import CLIPTokenizer

from taiyi.drawer.TaiyiDrawer import TaiyiDrawer


class TaiyiMetaDrawer:
    def __init__(self, draw_meta: dict):
        self.__draw_meta = draw_meta

        # CLIP length limitation truncate workaround
        self.__long_prompt_arg = 1
        if self.__draw_meta.__contains__('long_prompt_arg'):
            self.__long_prompt_arg = self.__draw_meta['long_prompt_arg']

        # whether turn off NSFW check
        self.__turn_off_nsfw_check = False
        if self.__draw_meta.__contains__('turn_off_nsfw_check'):
            self.__turn_off_nsfw_check = self.__draw_meta['turn_off_nsfw_check']

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
        if tokenizer is not None:
            params['tokenizer'] = tokenizer
        if self.__long_prompt_arg > 1:
            params['custom_pipeline'] = "lpw_stable_diffusion"

        if model_meta.__contains__('path'):
            drawer = TaiyiDrawer(model_meta['path'], **params)
        else:
            drawer = TaiyiDrawer(model_meta['name'], **params)

        if self.__turn_off_nsfw_check:
            drawer.turn_off_nsfw_check()

        self.__decide_scheduler(drawer)

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

        height = self.__draw_meta['height']
        width = self.__draw_meta['width']

        # 提示词
        prompt = self.__draw_meta['prompt']
        # 负面提示词
        negative_prompt = self.__draw_meta['negative_prompt']
        # 降噪步数 数字越大，时间越长
        num_inference_steps = self.__draw_meta['steps']
        # 遵循提示词的级别 数字越大越接近提示词，但图像质量会下降 CFG?
        guidance_scale = self.__draw_meta['cfg']

        if self.__long_prompt_arg > 1:
            drawn = drawer.draw(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                max_embeddings_multiples=self.__long_prompt_arg
            )
        else:
            drawn = drawer.draw(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            )

        image = drawn.images[0]
        if output_image_file is None:
            image.show()
        else:
            image.save(output_image_file)

        # filename = SampleConstant.output_dir + f"\\NijiGirl-{time.time()}.jpg"
        # image.save(filename)
        # print(f"saved: {filename}")
