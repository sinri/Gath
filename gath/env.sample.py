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
    'sd_v1_5': {
        'name': 'runwayml/stable-diffusion-v1-5'
    }
}
inn_textual_inversion_dict = {

}

inn_output_folder = ''
