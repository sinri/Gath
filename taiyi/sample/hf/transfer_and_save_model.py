from taiyi.drawer.TaiyiDrawer import TaiyiDrawer


class LoRAModelMerger:
    def __init__(self, base_model_dir: str, device: str = 'cuda'):
        self.__base_model_dir = base_model_dir
        self.__device = device

    def merge_lora_on_base(
            self,
            lora_model_safetensors_file,
            huggingface_repo_dir,
            multiplier: float = 1.0
    ):
        drawer = TaiyiDrawer(self.__base_model_dir)
        drawer.to_device(self.__device)
        drawer.load_lora_weights(lora_model_safetensors_file, multiplier, self.__device)
        drawer.save_model(huggingface_repo_dir)

    @staticmethod
    def merge_lxq(base_model_dir: str):
        LoRAModelMerger(base_model_dir).merge_lora_on_base(
            'E:\\sinri\\sd-scripts\\workspace\\lxq2\\trained_model\\stable-diffusion-v1-5-LeXiaoQi-3.safetensors',
            'E:\\OneDrive\\Leqee\\ai\\stable-diffusion-v1-5-LeXiaoQi'
        )

    @staticmethod
    def merge_anything_v5(base_model_dir: str):
        LoRAModelMerger(base_model_dir).merge_lora_on_base(
            'E:\\OneDrive\\Leqee\\ai\\civitai\\AnythingV5_v32.safetensors',
            'E:\\OneDrive\\Leqee\\ai\\AnythingV5'
        )


if __name__ == '__main__':
    base_model = "E:\\sinri\\stable-diffusion-webui\\models\\Stable-diffusion\\stable-diffusion-v1-5"
    # lora_model = 'E:\\sinri\\sd-scripts\\workspace\\lxq2\\trained_model\\stable-diffusion-v1-5-LeXiaoQi-3.safetensors'
    # repo = 'E:\\OneDrive\\Leqee\\ai\\stable-diffusion-v1-5-LeXiaoQi'

    LoRAModelMerger.merge_anything_v5(base_model)

    print('喵喵喵')
