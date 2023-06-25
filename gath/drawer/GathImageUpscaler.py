import torch
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline


class GathImageUpscaler:
    def __init__(self,upscaler:StableDiffusionLatentUpscalePipeline):
        # model_id = "stabilityai/sd-x2-latent-upscaler"
        # upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        # upscaler.to("cuda")
        self.__upscaler=upscaler

    def refine(self,pipeline:StableDiffusionPipeline,low_res_latents,output_file_path:str):
        # low_res_latents = pipeline(prompt, generator=generator, output_type="latent").images

        with torch.no_grad():
            image = pipeline.decode_latents(low_res_latents)
        #image = pipeline.numpy_to_pil(image)[0]
        #image.save(output_file_path)

        upscaled_image = self.__upscaler(
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
        ).images[0]

        upscaled_image.save("../images/a2.png")