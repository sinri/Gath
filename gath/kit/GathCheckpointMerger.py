from diffusers import StableDiffusionPipeline


class GathCheckpointMerger:
    """
    CIVITAI 有很多奇奇怪怪的ckpt，只要网络好，就可以还原回repo，妙极了
    """

    def __init__(self, pretrained_model_link_or_path, **kwargs):
        self.__stable_diffusion = StableDiffusionPipeline.from_ckpt(pretrained_model_link_or_path, **kwargs)

    def save_model(self, save_directory: str):
        # self.__stable_diffusion('a girl and a cat')
        self.__stable_diffusion.save_pretrained(save_directory, True)


if __name__ == '__main__':
    ckpt_path = 'E:\\OneDrive\\Leqee\\ai\\civitai\\ckpt_AbyssOrangeMix2\\abyssorangemix2NSFW_abyssorangemix2Nsfw.safetensors'
    repo_path = 'E:\\OneDrive\\Leqee\\ai\\repo\\AbyssOrangeMix2-NSFW'

    x = GathCheckpointMerger(ckpt_path)
    x.save_model(repo_path)
