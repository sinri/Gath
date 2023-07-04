from gath.kit.GathCheckpointMerger import GathCheckpointMerger


def convert(ckpt_path, repo_path):
    x = GathCheckpointMerger(ckpt_path)
    x.save_model(repo_path)


if __name__ == '__main__':
    convert(
        'E:\\OneDrive\\Leqee\\ai\\civitai\\ckpt_Hassaku\\hassakuHentaiModel_v13.safetensors',
        'E:\\OneDrive\\Leqee\\ai\\repo\\HassakuHentai'
    )
