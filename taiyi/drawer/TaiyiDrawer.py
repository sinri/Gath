from collections import defaultdict
from typing import Union, List

import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from safetensors.torch import load_file


class TaiyiDrawer:
    def __init__(self, vae: Union[AutoencoderKL, str], **kwargs):
        """
        Initialize TaiyiDrawer along with its pipeline.
        :param vae:  Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        Others in kwargs:
            text_encoder (CLIPTextModel) — Frozen text-encoder. Stable Diffusion uses the text portion of CLIP, specifically the clip-vit-large-patch14 variant.
            tokenizer (CLIPTokenizer) — Tokenizer of class CLIPTokenizer.
            unet (UNet2DConditionModel) — Conditional U-Net architecture to denoise the encoded image latents.
            scheduler (SchedulerMixin) — A scheduler to be used in combination with unet to denoise the encoded image latents. Can be one of DDIMScheduler, LMSDiscreteScheduler, or PNDMScheduler.
            safety_checker (StableDiffusionSafetyChecker) — Classification module that estimates whether generated images could be considered offensive or harmful. Please, refer to the model card for details.
            feature_extractor (CLIPImageProcessor) — Model that extracts features from generated images to be used as inputs for the safety_checker.
        """

        # if kwargs.__contains__('tokenizer'):
        # CLIPTokenizer(vae)

        self.__stable_diffusion = StableDiffusionPipeline.from_pretrained(vae, **kwargs)

        # print(self.__stable_diffusion.tokenizer)

    def to_device(self, device):
        """
        Change device, so you may use CUDA.
        :param device: such as `cuda`
        :return:
        """
        self.__stable_diffusion = self.__stable_diffusion.to(device)
        return self

    def load_textual_inversion(self, pretrained_model_name_or_path):
        """
        :param pretrained_model_name_or_path: The embedding or hyper network file (.pt)
        :return:
        """
        self.__stable_diffusion.load_textual_inversion(pretrained_model_name_or_path)
        return self

    def load_lora_weights(self, checkpoint_path):
        """
        https://github.com/huggingface/diffusers/issues/3064#issuecomment-1510653978
        USAGE:
        lora_model = lora_models + "/" + opt.lora + ".safetensors"
        self.pipe = load_lora_weights(self.pipe, lora_model)
        """
        # First, load base model
        # You should USE CUDA.
        # self.__stable_diffusion.to("cuda")
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        alpha = 0.75
        # load LoRA weight from .safetensors
        state_dict = load_file(checkpoint_path, device="cuda")
        visited = []

        # directly update weight in diffusers model
        for key in state_dict:
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            # as we have set the alpha beforehand, so just skip
            if ".alpha" in key or key in visited:
                continue

            if "text" in key:
                layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = self.__stable_diffusion.text_encoder
            else:
                layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = self.__stable_diffusion.unet

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

            pair_keys = []
            if "lora_down" in key:
                pair_keys.append(key.replace("lora_down", "lora_up"))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace("lora_up", "lora_down"))

            # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)
                curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

            # update visited list
            for item in pair_keys:
                visited.append(item)

        return self

    def load_lora_weights_v2(self, checkpoint_path, multiplier, device, dtype):
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
                curr_layer = self.__stable_diffusion.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = self.__stable_diffusion.unet

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
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2),
                                                                        weight_down.squeeze(3).squeeze(2)).unsqueeze(
                    2).unsqueeze(3)
            else:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

        return self

    @staticmethod
    def build_dummy_safety_checker():
        return lambda images, clip_input: (images, False)

    def turn_off_nsfw_check(self):
        """
        https://github.com/CompVis/stable-diffusion/issues/239#issuecomment-1241838550
        """
        self.__stable_diffusion.safety_checker = self.build_dummy_safety_checker()
        return self

    def change_scheduler(self, new_scheduler):
        """
        DDIMScheduler
        DPMSolverMultistepScheduler (別名 DPM-Solver++ diffusers v0.8.0~)
        LMSDiscreteScheduler
        PNDMScheduler
        EulerDiscreteScheduler　(diffusers v0.7.0~)
        EulerAncestralDiscreteScheduler (略称 Euler a, diffusers v0.7.0~)

        デフォルトではPNDMSchedulerになっているそうです。
        また、0.7.0で追加されたEuler系Schedulerは計算速度が高速で、20~30ステップでも良い結果を出力してくれるそうです。

        :param new_scheduler:
        :return:
        """
        self.__stable_diffusion.scheduler = new_scheduler
        return self

    def change_scheduler_to_euler_discrete(self):
        return EulerDiscreteScheduler.from_config(self.__stable_diffusion.scheduler.config)

    def change_scheduler_to_euler_ancestral_discrete(self):
        return EulerAncestralDiscreteScheduler.from_config(self.__stable_diffusion.scheduler.config)

    def draw(self, prompt: Union[str, List[str]], **kwargs):
        """
        Draw an image
        :param prompt: The prompt or prompts to guide the image generation. If not defined, one has to pass prompt_embeds. instead.
        Others in kwargs:
            height (int, optional, defaults to self.unet.config.sample_size * self.vae_scale_factor) — The height in pixels of the generated image.
            width (int, optional, defaults to self.unet.config.sample_size * self.vae_scale_factor) — The width in pixels of the generated image.
            num_inference_steps (int, optional, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            guidance_scale (float, optional, defaults to 7.5) — Guidance scale as defined in Classifier-Free Diffusion Guidance. guidance_scale is defined as w of equation 2. of Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1. Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            negative_prompt (str or List[str], optional) — The prompt or prompts not to guide the image generation. If not defined, one has to pass negative_prompt_embeds instead. Ignored when not using guidance (i.e., ignored if guidance_scale is less than 1).
            num_images_per_prompt (int, optional, defaults to 1) — The number of images to generate per prompt.
            eta (float, optional, defaults to 0.0) — Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to schedulers.DDIMScheduler, will be ignored for others.
            generator (torch.Generator or List[torch.Generator], optional) — One or a list of torch generator(s) to make generation deterministic.
            latents (torch.FloatTensor, optional) — Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random generator.
            prompt_embeds (torch.FloatTensor, optional) — Pre-generated text embeddings. Can be used to easily tweak text inputs, e.g. prompt weighting. If not provided, text embeddings will be generated from prompt input argument.
            negative_prompt_embeds (torch.FloatTensor, optional) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, e.g. prompt weighting. If not provided, negative_prompt_embeds will be generated from negative_prompt input argument.
            output_type (str, optional, defaults to "pil") — The output format of the generate image. Choose between PIL: PIL.Image.Image or np.array.
            return_dict (bool, optional, defaults to True) — Whether or not to return a StableDiffusionPipelineOutput instead of a plain tuple.
            callback (Callable, optional) — A function that will be called every callback_steps steps during inference. The function will be called with the following arguments: callback(step: int, timestep: int, latents: torch.FloatTensor).
            callback_steps (int, optional, defaults to 1) — The frequency at which the callback function will be called. If not specified, the callback will be called at every step.
            cross_attention_kwargs (dict, optional) — A kwargs dictionary that if specified is passed along to the AttentionProcessor as defined under self.processor in diffusers.cross_attention.
        :return:
        """

        return self.__stable_diffusion(prompt, **kwargs)


if __name__ == '__main__':
    model_name = 'IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'
    model_dir = "E:\\sinri\\Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
    # Fetched Model Manually Before Running
    d = TaiyiDrawer(model_dir)
    # Download Model Automatically (SLOW)
    # d=TaiyiDrawer(model_name)

    # 提示词
    prompt = "三号桌子,白色桌布,玻璃花瓶里插着白百合"
    # 降噪步数 数字越大，时间越长
    num_inference_steps = 10
    # 遵循提示词的级别 数字越大越接近提示词，但图像质量会下降
    guidance_scale = 5
    # 负面提示词
    negative_prompt = "大圆桌"

    drawn = d.draw(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt

    )
    image = drawn.images[0]
    image.show()
