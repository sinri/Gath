# MODEL might be HuggingFace model id or downloaded repo path
sd_v1_5_model = 'runwayml/stable-diffusion-v1-5'

output_dir = '../output'

inn_token = 'SomeToken'
mysql_config = {
    'host': '',
    'port': 3306,
    'user': '',
    'password': '',
    'db': '',
    'charset': 'utf8',
    'auto_commit': True,
}

inn_base_meta = {
    'generator': {
        'framework': 'torch',
    },
    'turn_off_nsfw_check': 'true',
    # model
    # tokenizer
    # textual_inversion
    'device': 'cuda',
    'height': 512,
    'width': 512,
    'long_prompt_arg': 1,
    # lora
    # prompt
    # negative_prompt
    'steps': 20,
    'cfg': 7.5,
    'scheduler': 'euler_a',
    # seed
}


inn_model_dict = {
    'sd-1.5': {
        'name': 'runwayml/stable-diffusion-v1-5',
        'path': 'E:\\sinri\\stable-diffusion-webui\\models\\Stable-diffusion\\stable-diffusion-v1-5',
        'tokenizer': {
            'name': 'unknown',
            'path': 'E:\\sinri\\stable-diffusion-webui\\models\\Stable-diffusion\\stable-diffusion-v1-5\\tokenizer',
            'max_length': 1024,
        }
    },
    'sd-1.5-LeXiaoQi': {
        'name': 'ljni/stable-diffusion-v1-5-LeXiaoQi',
        'path': 'E:\\OneDrive\\Leqee\\ai\\stable-diffusion-v1-5-LeXiaoQi',
        'tokenizer': {
            'name': 'unknown',
            'path': 'E:\\OneDrive\\Leqee\\ai\\stable-diffusion-v1-5-LeXiaoQi\\tokenizer',
            'max_length': 1024,
        }
    },
    'waifu': {
        'path': 'E:\\OneDrive\\Leqee\\ai\\waifu-diffusion',
        'tokenizer': {
            'path': 'E:\\OneDrive\\Leqee\\ai\\waifu-diffusion\\tokenizer',
            'max_length': 1024,
        }
    },
    'openjourney': {
        'path': 'E:\\OneDrive\\Leqee\\ai\\openjourney',
        'tokenizer': {
            'path': 'E:\\OneDrive\\Leqee\\ai\\openjourney\\tokenizer',
            'max_length': 1024,
        }
    },
    'a-certain': {
        'path': 'E:\\OneDrive\\Leqee\\ai\\ACertainModel',
        'tokenizer': {
            'path': 'E:\\OneDrive\\Leqee\\ai\\ACertainModel\\tokenizer',
            'max_length': 1024,
        }
    },
    'anything-5': {
        'type': 'ckpt',
        'path': 'E:\\OneDrive\\Leqee\\ai\\civitai\\AnythingV5_v32.safetensors',
    },
}
inn_textual_inversion_dict = {

}

inn_output_folder = ''

inn_oss_ak_id = ''
inn_oss_ak_secret = ''
inn_oss_bucket = ''
inn_oss_endpoint = 'https://oss-cn-hangzhou.aliyuncs.com'
