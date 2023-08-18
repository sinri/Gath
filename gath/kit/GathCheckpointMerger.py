from diffusers import StableDiffusionPipeline


class GathCheckpointMerger:
    """
    CIVITAI 有很多奇奇怪怪的ckpt，只要网络好，就可以还原回repo，妙极了
    """

    def __init__(self, pretrained_model_link_or_path, **kwargs):
        self.__stable_diffusion = StableDiffusionPipeline.from_single_file(pretrained_model_link_or_path, **kwargs)

    def save_model(self, save_directory: str):
        # self.__stable_diffusion('a girl and a cat')
        self.__stable_diffusion.save_pretrained(save_directory, True)


if __name__ == '__main__':
    # ckpt_path = 'E:\\OneDrive\\Leqee\\ai\\civitai\\ckpt_only\\onlyrealistic_v30BakedVAE.safetensors'
    ckpt_path = 'E:\\OneDrive\\Leqee\\ai\\TosiArt\\ckpt_LoliStyleMixS\\LoliStyle-Mix-S.ckpt'
    repo_path = 'E:\\sinri\\HuggingFace\\LoliStyle-Mix-S'

    x = GathCheckpointMerger(ckpt_path)
    x.save_model(repo_path)
