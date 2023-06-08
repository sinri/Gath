from taiyi.drawer.LoRAModelMerger import LoRAModelMerger


def merge_lxq(base_model_dir: str):
    LoRAModelMerger(base_model_dir).merge_lora_on_base(
        'E:\\sinri\\sd-scripts\\workspace\\lxq2\\trained_model\\stable-diffusion-v1-5-LeXiaoQi-3.safetensors',
        'E:\\OneDrive\\Leqee\\ai\\stable-diffusion-v1-5-LeXiaoQi'
    )


def merge_anything_v5(base_model_dir: str):
    LoRAModelMerger(base_model_dir).merge_lora_on_base(
        'E:\\OneDrive\\Leqee\\ai\\civitai\\AnythingV5_v32.safetensors',
        'E:\\OneDrive\\Leqee\\ai\\AnythingV5'
    )


if __name__ == '__main__':
    base_model = "E:\\sinri\\stable-diffusion-webui\\models\\Stable-diffusion\\stable-diffusion-v1-5"
    # lora_model = 'E:\\sinri\\sd-scripts\\workspace\\lxq2\\trained_model\\stable-diffusion-v1-5-LeXiaoQi-3.safetensors'
    # repo = 'E:\\OneDrive\\Leqee\\ai\\stable-diffusion-v1-5-LeXiaoQi'

    merge_anything_v5(base_model)

    print('喵喵喵')
