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
    # ckpt_path = 'E:\\sinri\\stable-diffusion-webui\\models\\Stable-diffusion\\anything-v5-0.7.safetensors'
    # repo_path = 'E:\\OneDrive\\Leqee\\ai\\AnythingV5'

    # ckpt_path = 'E:\\OneDrive\\Leqee\\ai\\civitai\\ICBINP\\icbinpICantBelieveIts_final.safetensors'
    # repo_path = 'E:\\OneDrive\\Leqee\\ai\\ICBINP'

    # ckpt_path = 'E:\\OneDrive\\Leqee\\ai\\civitai\\DisneyPixarCartoon\\disneyPixarCartoon_v10.safetensors'
    # repo_path = 'E:\\OneDrive\\Leqee\\ai\\DisneyPixarCartoon'

    # ckpt_path = 'E:\\OneDrive\\Leqee\\ai\\civitai\\NightSkyYozoraStyleModel\\nightSkyYOZORAStyle_yozoraV1PurnedFp16.safetensors'
    # repo_path = 'E:\\OneDrive\\Leqee\\ai\\NightSkyYozoraStyleModel'

    # ckpt_path = 'E:\\OneDrive\\Leqee\\ai\\civitai\\ckpt_RealDosMix\\realdosmix_.safetensors'
    # repo_path = 'E:\\OneDrive\\Leqee\\ai\\repo\\RealDosMix'

    ckpt_path = 'E:\\OneDrive\\Leqee\\ai\\civitai\\vae_BerrysMix\\BerrysMix.vae.safetensors'
    repo_path = 'E:\\OneDrive\\Leqee\\ai\\repo\\BerrysMixVae'

    x = GathCheckpointMerger(ckpt_path)
    x.save_model(repo_path)
