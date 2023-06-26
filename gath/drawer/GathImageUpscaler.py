import torch
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline

from gath import env


class GathImageUpscaler:
    def __init__(self, upscaler: StableDiffusionLatentUpscalePipeline):
        # model_id = "stabilityai/sd-x2-latent-upscaler"
        # upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        # upscaler.to("cuda")
        self.__upscaler = upscaler


if __name__ == '__main__':
    model = 'E:\\OneDrive\\Leqee\\ai\\repo\\DisneyPixarCartoon'
    # model="CompVis/stable-diffusion-v1-4"
    pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipeline.to("cuda")

    upscaler_model = 'E:\\OneDrive\\Leqee\\ai\\repo\\sd-x2-latent-upscaler'
    # upscaler_model="stabilityai/sd-x2-latent-upscaler"
    upscaler: StableDiffusionLatentUpscalePipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(
        upscaler_model,
        torch_dtype=torch.float16)
    upscaler.to("cuda")

    prompt = "a photo of an astronaut, high resolution, unreal engine, ultra realistic"
    generator = torch.Generator('cuda')
    generator.manual_seed(33)

    # we stay in latent space! Let's make sure that Stable Diffusion returns the image
    # in latent space
    low_res_latents = pipeline(
        prompt,
        negative_prompt='blurry',
        num_inference_steps=20,
        guidance_scale=7,
        generator=generator,
        output_type="latent"
    ).images

    print(type(low_res_latents))

    upscaled_image = upscaler(
        prompt=prompt,
        image=low_res_latents,
        num_inference_steps=20,
        guidance_scale=0,
        generator=generator,
    ).images[0]

    # Let's save the upscaled image under "upscaled_astronaut.png"
    upscaled_image.save(f"{env.output_dir}/astronaut_1024.png")

    # as a comparison: Let's also save the low-res image
    with torch.no_grad():
        image = pipeline.decode_latents(low_res_latents)
    image = pipeline.numpy_to_pil(image)[0]

    image.save(f"{env.output_dir}/astronaut_512.png")
