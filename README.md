# TaiyiDrawer
Taiyi Drawer Related

## Draw

Tested under:
* CPU: 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz   3.50 GHz
* Memory: 64GB 
* Windows 10 企业版 19044.1889
* NVIDIA GeForce RTX 3060

Directly use the model of [IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1),
or fetch the trained files to local with

```commandline
git lfs install
git clone https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1
```

Then use [TaiyiDrawer](./taiyi/drawer/TaiyiDrawer.py) to draw.
It is based on HuggingFace Pipeline.
See its MAIN for more information.