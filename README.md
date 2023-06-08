# GathDrawer

Gath Drawer Related

## Draw

Tested under:

* CPU: 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz 3.50 GHz
* Memory: 64GB
* Windows 10 企业版 19044.1889
* NVIDIA GeForce RTX 3060

Directly use the model
of [IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1),
or fetch the trained files to local with

```commandline
git lfs install
git clone https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1
```

Then use [GathDrawer](gath/drawer/GathDrawer.py) to draw.
It is based on HuggingFace Pipeline.
See its MAIN for more information.

## Prompt Engineering

[魔法导论必备工具, 简单易用的AI绘画tag生成器](https://aitag.top/)  By bilibili 波西BrackRat

* 使用 `,` 进行提示词分割
* 自然语言，最好是英语，是可行的，但是理解能力有限
* 最好使用标签
  * 标签可以使用权重
  * 默认情况下越靠前的提示词权重越高
  * 通过 (提示词:权重数值) 手动设置权重，比如： (1cat:1.3),(1dog:0.8)
  * 通过 `()` `{}` `[]` 设置权重：
    - `(提示词)` 等价于 (提示词:1.1)
    - `{提示词}` 等价于 (提示词:1.05)
    - `[提示词]` 等价于 (提示词:0.952) 即 1/1.05
    - 且 `()` `{}` `[]` 语法可嵌套使用，比如 (((提示词))) 就等价于 (提示词:1.331)。
  * 一般情况下建议使用 (提示词:权重数值) 语法，可读性、可控性更高。
  * 注意一般情况下权重不建议超过 1.5，不然会对画面造成巨大影响。
* 除了基础提示语外，类似于 Lora 模型也是需要使用提示语来饮用的，语法：`<lora:模型⽂件名:权重>`
  * 比如如果要使用知名的模型墨心，提示词是这样的 `<lora:MoXinV1:1>`
* 进阶语法
  * OR
    * 比如在绘制头发时通过 `[purple|sliver|green]_hair` 可以绘制出这样的混色的发色
    * 也可以搭配 `multicolor hair` 生成
  * AND
    * `purple hair AND sliver hair AND green hair`
    * AND 语法还支持为某个片段增加权重，比如 `gold hair :1.2 AND sliver hair :0.8 AND green hair` 可以让发色更多金色
  * 步骤控制语法
    * 比如 `[cat:10]` 指从第十步开始画猫
    * 而 `[cat::20]` 表示在第二十步结束画猫
    * 也可以组合使用，比如： `[[cat::20]:10]` 代表从第十步开始第二十步结束。
  * 关键字占比控制
    * 比如 [dog:girl:0.9] 表示总绘制步骤的前 90% 画狗，后面画女孩
    * 而 [dog:girl:30] 则表示前三十步画狗，后面画女孩

## Embeddings

[With WebUI](https://www.bilibili.com/read/cv20183008?from=search)