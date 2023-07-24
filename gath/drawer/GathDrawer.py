import os
from typing import Union, List, Optional

from diffusers import StableDiffusionPipeline, StableDiffusionLatentUpscalePipeline, SchedulerMixin

from gath.drawer.GathDrawKit import GathDrawKit


class GathDrawer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        """
        :see diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
        :param pretrained_model_name_or_path:
        :param kwargs:
        :return:
        """
        x = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return GathDrawer(x)

    @classmethod
    def from_ckpt(cls, pretrained_model_link_or_path, **kwargs):
        """
        It would cause a download for the base model.
        see diffusers.loaders.FromCkptMixin.from_ckpt
        :param pretrained_model_link_or_path:
        :param kwargs:
        :return:
        """
        d = StableDiffusionPipeline.from_ckpt(pretrained_model_link_or_path, **kwargs)
        return GathDrawer(d)

    def __init__(self, pipeline: StableDiffusionPipeline):
        self.__pipeline = pipeline
        self.__upscaler: Optional[StableDiffusionLatentUpscalePipeline] = None

    def to_device(self, device):
        """
        Change device, so you may use CUDA.
        :param device: such as `cuda`
        :return:
        """
        self.__pipeline = self.__pipeline.to(device)
        return self

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

    def change_scheduler(self, new_scheduler: SchedulerMixin):
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
        self.__pipeline.scheduler = new_scheduler
        return self

    def change_scheduler_to_euler_discrete(self):
        GathDrawKit.change_scheduler_to_euler_discrete(self.__pipeline)
        return self

    def change_scheduler_to_euler_ancestral_discrete(self):
        GathDrawKit.change_scheduler_to_euler_ancestral_discrete(self.__pipeline)
        return self

    def change_scheduler_to_ddim(self):
        GathDrawKit.change_scheduler_to_ddim(self.__pipeline)
        return self

    def change_scheduler_to_dpm_solver_multistep(self):
        GathDrawKit.change_scheduler_to_dpm_solver_multistep(self.__pipeline)
        return self

    def change_scheduler_to_lms_discrete(self):
        GathDrawKit.change_scheduler_to_lms_discrete(self.__pipeline)
        return self

    def change_scheduler_to_pndm(self):
        GathDrawKit.change_scheduler_to_pndm(self.__pipeline)
        return self

    def change_scheduler_to_uni_pc_multistep(self):
        GathDrawKit.change_scheduler_to_uni_pc_multistep(self.__pipeline)
        return self

    def change_scheduler_to_kdpm_2_discrete(self):
        GathDrawKit.change_scheduler_to_kdpm_2_discrete(self.__pipeline)
        return self

    def set_upscaler(self, upscaler: StableDiffusionLatentUpscalePipeline):
        self.__upscaler = upscaler
        return self

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

        if self.__upscaler is not None:
            # print('upscaler is ready')

            kwargs['output_type'] = "latent"
            low_res_latents = self.__pipeline(prompt, **kwargs).images

            # print(type(low_res_latents))

            # return self.__upscale(prompt, low_res_latents, 20, 0, kwargs['generator'])

            return self.__upscaler(
                prompt=prompt,
                image=low_res_latents,
                num_inference_steps=20,
                guidance_scale=0,
                generator=kwargs['generator'],
            )
        else:
            return self.__pipeline(prompt, **kwargs)

    def __upscale(self, prompt, low_res_latents, steps, guidance_scale, generator):
        return self.__upscaler(
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    def save_model(self, save_directory: str):
        self.__pipeline.save_pretrained(save_directory, True)


if __name__ == '__main__':
    model_name = 'IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'
    model_dir = "E:\\sinri\\Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
    # Fetched Model Manually Before Running
    d = GathDrawer.from_pretrained(model_dir)
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
