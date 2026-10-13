# v2 卡通素材与 2D 动画制作

本方案使用小型 Transformer 将中文提示词转换为几何绘图指令，再用 Pillow 渲染为 PNG。支持动物、可变卡通人物和常用环境元素，仍然是有限模板方案，不是通用文生图模型。

新增动画层将角色外观固定下来，用连续关节姿态、表情参数和显式时间轴控制运动。**动画不是逐帧独立推理，也不需要先训练模型才能预览。** 当前适合验证镜头、角色和动作的简化 2D 动画原型，不是完整动画片自动生成系统。

## 一键用模型素材制作 MP4

训练好的 `tiny_model.pt` 放在 `v2` 中后，运行 `make_film.py`。它只加载一次模型，依次生成女孩、小猫、郁金香、树、房屋、太阳和云的透明素材，再合成三个镜头：公园里相遇、小猫发现花、一起分享美好的一天。

首次使用安装视频编码依赖（如已有系统 FFmpeg，可以不安装 `imageio-ffmpeg`）：

```powershell
Set-Location C:\Workspace\GenImg\v2
python -m pip install imageio-ffmpeg
python make_film.py
```

默认生成 **18 秒、640 × 360、24 fps、无音轨的 H.264 MP4**。输出位于脚本旁带时间戳的 `film_...` 新目录中；运行结束会打印完整视频路径。模型推理使用 CPU，生成素材期间可能需要等待，不会重新训练或覆盖模型。

输出包含：

- `film.mp4`：可以直接播放的短片。
- `assets/`：每个素材的 PNG 和绘图指令；所有镜头复用同一张角色图片。
- `assets/manifest.json`：提示词、图片尺寸、模型生成或模板回退的来源及警告。
- `background.png`：静态背景。
- `story.json`：镜头顺序、实际帧数、时长和素材来源。
- `encoder.log`：视频编码日志。

指定输出目录、尺寸和速度，或保留全部 PNG 帧：

```powershell
python make_film.py --output my-short-film --width 960 --height 540 --fps 24 --seconds-per-shot 6 --save-frames
```

`--seconds-per-shot` 设置每个镜头的时长，总共三个镜头；实际时长按帧数取整。宽高必须为偶数。输出目录必须不存在，防止覆盖已有作品。加 `--save-frames` 后会额外保存 `frames/frame_000000.png` 起的编号帧。

默认情况下，模型指令失败时会使用现有推理程序的模板回退，并在终端和素材清单中明确标记；不能把回退素材算作模型生成成功。要求全部素材必须来自模型时使用：

```powershell
python make_film.py --strict-model
```

严格模式遇到任何回退就停止，已生成的素材仍保留供检查。素材生成通过语法检查不代表外观一定符合提示词，仍应查看成片。

**这个入口实际使用模型输出的图片，但运动是固定图片的位移、轻微旋转、跳跃与镜头切换，不是骨骼行走、动态手势或口型同步。** 与下面 `animate.py` 的模板关节动画入口不同，它不会用模板动画替换模型角色图片。它也不调用万相模型，不自动生成配音或音乐。

已验证的示例成片：[film-preview/film.mp4](film-preview/film.mp4)。该次七个素材中六个来自模型，郁金香回退到模板，具体来源见 [film-preview/assets/manifest.json](film-preview/assets/manifest.json)。

## 动画制作入口

在 `v2` 目录运行多角色示例：

```powershell
Set-Location C:\Workspace\GenImg\v2
python animate.py --scene scene.example.json --output animation_out --gif
```

示例镜头包含固定天空、山和房屋，摇摆的树，移动后挥手并切换表情的女孩，以及走动的小猫。导出 4 秒、12 fps、640 × 360 的 48 张 PNG。输出目录必须是新目录或空目录，程序不会静默覆盖已有帧。

单个角色也可以直接预览：

```powershell
python animate.py --prompt "女孩站着 马尾 蓝色长袖 红色长裤 张开手" --action wave --duration 3 --fps 24 --output girl-wave --gif
python animate.py --prompt "橘色猫站着" --action run --output cat-run --gif
python animate.py --prompt "绿色阔叶树" --action wind --output tree-wind --gif
```

每个输出目录包含：

- `frame_000000.png` 起的编号 RGBA 帧，用于合成或导入剪辑软件。
- `background.png`：渲染一次的背景快照；动态植物应放在 `actors`，不能放在背景项中。
- `scene.json`：保存完整镜头和解析后的角色属性，后续重载无需重新解析原始提示词。
- `metadata.json`：帧数、fps、固定外观摘要、背景摘要和角色属性。
- `preview.gif`：仅在 `--gif` 时生成，最多约 12 fps、180 帧、640 × 360。GIF 会采样和量化颜色，完整质量以 PNG 为准；预览不依赖 FFmpeg。

重载已冻结的镜头：

```powershell
python animate.py --scene animation_out\scene.json --output animation_reload --gif
```

### 场景与时间轴

参考 [scene.example.json](scene.example.json)。场景固定 `version: 1`、`width`、`height`、`fps` 和 `duration`；角色尺寸和锚点不会按每帧内容重新缩放或居中。每帧时间为 `帧号 / fps`，不重复输出末帧。

| 字段 | 用途 |
| --- | --- |
| `characters` | 以角色 ID 保存 `prompt`，或使用导出后保存的 `attributes`；同一 ID 复用同一外观 |
| `background.color` | 底色，支持 `transparent` |
| `background.ground_color`、`horizon` | 地面颜色与地平线高度比例 |
| `background.image` | 相对于场景文件的背景图片路径，尺寸必须与场景一致 |
| `background.items` | 固定素材列表，只接受静态动作，按数组顺序叠加 |
| `actors` | 动态或静态对象，按数组顺序叠加；背景始终在它们后面 |
| `actor.id` | 唯一对象 ID；影响确定性的自动眨眼时机 |
| `actor.character` 或 `prompt` | 引用角色 ID 或直接给出提示词，二选一 |
| `actor.position` | 场景像素中的 `[x, y]` 锚点；人物参考双脚中心，动物参考脚下，植物参考根部 |
| `actor.height` | 逻辑画布换算后的像素高度，**不是每帧可见包围盒高度**；另有固定留白防止跳跃和头发裁切 |
| `actor.action` | 默认动作；`static` 保留提示词里的静态姿态 |
| `actor.rate`、`phase` | 每秒循环次数和初始周期相位 |
| `actor.flip` | 固定水平镜像，布尔值；不是三维转身 |
| `actor.blink` | 是否自动眨眼，人物有效；设为 `false` 可检查精确循环首尾 |
| `actor.path` | `time`、`position` 关键点；从 0 开始，严格递增，线性移动，末点之后保持位置 |
| `actor.clips` | `start`、`end`、`action`、可选 `expression`、`rate`、`transition`；必须连续、无重叠且覆盖整个镜头 |

切换动作时 `transition` 表示新片段开头的渐变时长，默认最多 0.15 秒，可设为 0 禁用。人物跑、走、跳、挥手、呼吸采用固定长度的两段式关节解算；这些动作间用角度插值，避免直接插值坐标导致肢体缩短。原有坐、蹲、抱臂、叉腰是手工静态姿态，与它们的过渡会平滑插值原有比例，不是完整骨骼绑定系统。手势、衣裤鞋颜色、五官类型等来自固定角色属性；需要展开手指时在角色提示词里明确写 `张开手`。

| 对象 | 动画动作 |
| --- | --- |
| 人物 | `idle` / `stand`、`walk`、`run`、`jump`、`wave`；`sit`、`crouch`、`arms_crossed`、`hands_on_hips` 保持相应静态姿态；`static` 保留角色初始姿态 |
| 动物 | `idle`、`walk`、`run`、`jump`、`static`；简单交替迈步，曲线尾巴摆动，短尾跟随身体 |
| 树、草、花 | `wind`、`static`；上部摆动、根部固定的简化形变 |
| 其他元素 | `static`；仍可用 `path` 进行整体移动 |

表情支持开心、平静、难过、生气、惊讶、眨眼、苦笑，仅人物可指定。动作支持部分中文名，但建议在 JSON 中使用上表英文动作名。单角色动物提示词仍需带已有静态动作，例如 `橘色猫站着`，动画动作由 `--action` 或时间轴指定。

### 制作边界与训练顺序

1. 先用模板和 `animate.py` 检查角色外观、动作、配色和镜头；修改角色定义而不是逐帧改变外观。
2. 外观确认后，再统一运行 `python gen_data.py`、`python train.py`，让模型学习最新静态素材指令。
3. 动画由固定角色与时间轴驱动。重新训练静态模型不会自动增加转身视角、动作规律或互动能力；当前动画入口有意不加载模型。
4. 目前人物是正面卡通，跑走是简化原地步态加显式路径移动；不支持 360 度转身、自动打闹、碰撞、抓握、口型同步、自动机位或配音。跨角色互动须手工编排位置与动作，不能保证物理接触。

单次场景限制为 64 到 2048 像素边长、1 到 60 fps、0.1 到 120 秒、最多 3600 帧、32 个对象。建议按镜头拆分导出。位置可以在画面外，程序会按固定镜头正常裁切；GIF 不是透明视频交付格式。

安装 FFmpeg 后可把 PNG 序列转换为 MP4；默认流程无需安装它：

```powershell
ffmpeg -framerate 12 -i animation_out/frame_%06d.png -c:v libx264 -pix_fmt yuv420p animation.mp4
```

MP4 不保留透明背景，需要透明素材时保留 PNG 序列。奇数画面尺寸应先转换为适合编码器的偶数尺寸。

回归检查（不训练、不加载检查点）：

```powershell
python -m unittest test_genimg -v
```

## 1. 训练完成后快速出图

以下命令均在 Windows PowerShell 的 `v2` 目录中执行：

```powershell
Set-Location C:\Workspace\GenImg\v2
python infer.py "橘色猫站着" | python genimg.py --output cat.png --size 1024
```

图片保存在当前目录的 `cat.png`，输出大小为 1024 × 1024。当前训练模板使用透明背景，适合导入游戏引擎或图像编辑工具。

尚未使用最新数据重新训练时，请先加 `--template` 查看新人物和元素；现有模型不会自动学会新模板。

其他例子：

```powershell
python infer.py "蓝色兔子跳跃" | python genimg.py --output rabbit.png --size 512
python infer.py "棕色熊坐着" | python genimg.py --output bear.png --background "#ffffff"
python infer.py "粉色猪趴着" | python genimg.py --output pig.png --outline-width 0
```

### 训练后的人物提示词测试

以下命令不加 `--template`，用于检查训练完成的人物模型。均在 `v2` 目录中运行，每条命令使用不同的输出文件名，便于对比；重复执行同一条命令会覆盖对应图片。

先试身份、动作和表情：

```powershell
python infer.py "女孩站着" | python genimg.py --output girl-stand.png --size 512
python infer.py "男孩挥手" | python genimg.py --output boy-wave.png --size 512
python infer.py "开心的女孩挥手" | python genimg.py --output girl-happy-wave.png --size 512
python infer.py "平静的男人站着" | python genimg.py --output man-calm-stand.png --size 512
python infer.py "难过的女人坐着" | python genimg.py --output woman-sad-sit.png --size 512
python infer.py "开心的男孩奔跑" | python genimg.py --output boy-happy-run.png --size 512
python infer.py "惊讶的女孩跳跃" | python genimg.py --output girl-surprised-jump.png --size 512
python infer.py "生气的男人叉腰" | python genimg.py --output man-angry-hips.png --size 512
```

再试发型、衣物和独立配色：

```powershell
python infer.py "开心的女孩挥手，马尾，蓝色长袖，红色长裤" | python genimg.py --output girl-ponytail-wave.png --size 512
python infer.py "平静的男孩站着，短发，绿色短袖，灰色短裤" | python genimg.py --output boy-green-shirt.png --size 512
python infer.py "开心的女人站着，长发，红色外套，黑色长裤" | python genimg.py --output woman-red-jacket.png --size 512
python infer.py "生气的男人叉腰，短发，黑发，蓝色上衣，灰色裤子" | python genimg.py --output man-blue-shirt.png --size 512
```

提示词较短时，程序仍会补全默认外观，所以生成指令不一定更短。以上动作生成的是静态姿态图片，连续动画请使用 `animate.py`。

**检查终端提示，而不是只看图片是否正常。** 若出现“模型指令未完成或语法无效，已回退到模板”，说明模型输出未通过完整性或语法检查，最终图片由预设模板生成，不代表本次模型推理成功。没有回退提示只说明通过了这些检查，仍需观察人物身份、动作、表情和配色是否符合描述。记录出现回退的提示词，便于后续排查；不要仅因图片正常就忽略警告。

### 新版手脚和尾巴预览

新版模板使用带圆头、渐细的三次贝塞尔曲线绘制肢体和尾巴，并添加圆爪、趾线、猪蹄和鸭子蹼足。前肢姿态按动作调整，尾巴按物种区分。

无需重新训练即可预览模板造型：

```powershell
python infer.py --template "橘色猫跳跃" | python genimg.py --output cat-new.png
python infer.py --template "橘色狐狸站着" | python genimg.py --output fox-new.png
```

`--template` 明确绕过模型，不代表模型推理结果。原有模型仍可能生成旧直线指令；渲染器会给旧直线添加圆头，但不会自动把它们变成新版关节、爪子和尾巴。要让模型学习新版造型，需要重新运行 `python gen_data.py` 和 `python train.py`，完成后使用不带 `--template` 的普通生成命令。

造型预览见 `limbs-tail-preview.png`：从上到下为猫、狗、兔子、老鼠、熊、猪、狐狸、鸭子；从左到右为站着、坐着、跳跃、奔跑、趴着。

### 人物和常用元素预览

```powershell
python infer.py --template "开心的女孩挥手，长发，蓝色长袖，灰色长裤，白色运动鞋" | python genimg.py --output girl.png
python infer.py --template "生气的男人叉腰，深肤，黑发，外套，红色上衣，蓝色裤子，黑色靴子，壮实身材" | python genimg.py --output man.png
python infer.py --template "红色郁金香" | python genimg.py --output flower.png
python infer.py --template "绿色松树" | python genimg.py --output tree.png
python infer.py --template "蓝色小汽车" | python genimg.py --output car.png
python infer.py --template "黄色弯月" | python genimg.py --output moon.png
```

预览图：[人物](people-preview.png)、[常用元素](elements-preview.png)。人物和新元素使用 128 × 128 的逻辑画布，最终 PNG 尺寸仍由 `--size` 决定。

不要单独执行 `python genimg.py --output bear.png`：`--output` 只设置输出文件名，并不提供绘图指令。必须通过管道或 `--instructions` 提供输入。

## 2. 环境准备

需要 Python 3.9 或更新版本，以及 PyTorch、Pillow。建议使用独立虚拟环境，并在 VS Code 中选择该环境的解释器。

```powershell
python --version
python -m pip install torch Pillow
```

训练会在 CUDA 可用时使用 GPU，否则使用 CPU；当前推理脚本使用 CPU。GPU 加速需要与显卡驱动兼容的 PyTorch 安装版本。

## 3. 完整流程

```text
gen_data.py -> data.txt -> train.py -> tiny_model.pt
提示词 -> infer.py -> 绘图指令 -> genimg.py -> PNG
```

首次使用或修改训练模板后，按顺序执行：

```powershell
python gen_data.py
python train.py
python infer.py "橘色猫站着" | python genimg.py --output cat.png
```

- `gen_data.py` 生成共 3,230 条训练样本：400 条动物、310 条常用元素、2,520 条人物，覆盖已有的 `v2/data.txt`。
- `train.py` 训练结束时保存模型，覆盖已有的 `v2/tiny_model.pt`。模型不会在每轮训练后自动保存。
- 数据和模型文件路径以脚本所在目录为准；图片和手动指定的指令文件路径以终端当前目录为准。
- 已完成当前版本训练时，无需重复生成数据或训练，直接运行第 1 节的命令。

训练设置位于 `train.py` 顶部，默认 `EPOCHS = 160`、`BATCH_SIZE = 16`。修改数据、词表、模型结构或 tokenizer 后，需要重新训练。仅调整输出尺寸、背景或描边，无需重新训练。

人物数据覆盖 4 种人物 × 9 种动作 × 7 种表情 × 10 种上衣颜色，其余外观选项和衣裤鞋配色使用固定随机种子采样，并非所有属性的完整笛卡尔积。重复运行数据生成脚本会产生相同内容，便于复现实验。

训练序列长度自动从数据计算，不截断。相比旧 400 条短样本，训练时间和显存需求会增加。若显存不足，可将 `train.py` 中的 `BATCH_SIZE` 从 16 降到 4 或 2。无需先训练动物再训练人物，最新数据文件已经包含所有类别。

## 4. 提示词范围

`infer.py` 一次生成一个人物、动物或元素，不支持凭一句提示词自动组合多主体场景；多角色镜头使用 `animate.py` 的 JSON 场景。提示词通过关键词归一化到训练格式，不是开放式自然语言理解。

### 动物

建议使用明确的“颜色 + 动物 + 动作”。

| 类别 | 支持内容 |
| --- | --- |
| 动物 | 猫、狗、兔子、老鼠、熊、猪、狐狸、鸭子 |
| 动作 | 站着、坐着、跳跃、奔跑、趴着 |
| 颜色 | 橘色、黑色、白色、灰色、棕色、蓝色、红色、绿色、黄色、粉色 |

支持部分别名，例如“橙色”“站立”“跳起来”“跑步”“趴下”。不指定颜色时默认为橘色；动物和动作必须能被识别。

```powershell
python infer.py "一只可爱的橙色猫正在跳起来"
```

### 常用元素

可以只输入元素名称，或指定“颜色 + 造型”。不需要动作，未指定造型时使用表中的第一种。

| 元素 | 可选造型 | 默认颜色 |
| --- | --- | --- |
| 花 | 五瓣花、郁金香、雏菊 | 粉色 |
| 草 | 草丛、长草 | 绿色 |
| 树 | 阔叶树、松树、枯树 | 绿色 |
| 木 | 原木、木板 | 棕色 |
| 房屋 | 小屋、楼房、城堡 | 橘色 |
| 汽车 | 小汽车、货车 | 蓝色 |
| 河流 | 直河、弯河 | 蓝色 |
| 太阳 | 晴日、夕阳 | 黄色 |
| 月亮 | 弯月、满月 | 黄色 |
| 星星 | 五角星、四角星 | 黄色 |
| 云 | 积云、雨云 | 白色 |
| 山 | 青山、雪山 | 绿色 |
| 石头 | 圆石、岩石 | 灰色 |
| 蘑菇 | 圆顶蘑菇、斑点蘑菇 | 红色 |

“木头”“树木”“房子”“小河”“云朵”等常见别名可使用。颜色主要作用于主体，树干、花茎、轮胎、玻璃等部件保留各自的材质配色。

### 人物

人物按肩、肘、腕、髋、膝、踝的关节点绘制，衣物和皮肤覆盖相应肢段。五官包含眼白、瞳孔、高光、眼睑、眉毛、鼻子和嘴巴；手包含掌心、拇指及手指，鞋子包含鞋底和鞋带或靴筒。

| 属性 | 支持选项 | 未指定时 |
| --- | --- | --- |
| 人物类型 | 男孩、女孩、男人、女人 | 男孩；“女生”映射为女孩 |
| 动作 | 站着、坐着、奔跑、跳跃、挥手、行走、抱臂、叉腰、蹲下 | 站着 |
| 表情 | 开心、平静、难过、生气、惊讶、眨眼、苦笑 | 开心 |
| 发型 | 短发、长发、马尾、卷发 | 短发；女孩、女人默认马尾 |
| 发色 | 黑发、棕发、金发 | 棕发 |
| 肤色 | 浅肤、麦肤、深肤 | 浅肤 |
| 上衣 | 短袖、长袖、外套 | 短袖 |
| 裤子 | 长裤、短裤 | 长裤 |
| 鞋子 | 运动鞋、靴子 | 运动鞋 |
| 体型 | 普通身材、纤细身材、壮实身材 | 普通身材 |
| 眼睛 | 大眼睛、小眼睛 | 大眼睛 |
| 脸型 | 圆脸、长脸 | 圆脸 |
| 鼻子 | 小鼻子、圆鼻子 | 小鼻子 |
| 脖子 | 普通脖子、细脖子、粗脖子 | 普通脖子 |
| 手势 | 自然手、张开手、握拳 | 自然手；挥手、跳跃时自然张开 |
| 脚姿 | 自然脚姿、并脚、分脚 | 自然脚姿；并脚、分脚调整相对间距 |

上衣、裤子、鞋子可以分别使用前面的 10 种颜色，默认是蓝色上衣、灰色裤子、白色鞋子。请将颜色紧邻对应衣物名称，避免只列多个颜色而不写部件：

```text
女孩 挥手 开心 长发 蓝色长袖 红色短裤 白色运动鞋 张开手 细脖子
男人 行走 平静 短发 黑发 深肤 红色上衣 黑色长裤 棕色靴子 外套 壮实身材
男孩 跳跃 惊讶 卷发 金发 绿色短袖 蓝色短裤 白色运动鞋 大眼睛 圆鼻子
```

也支持“微笑”“高兴”“伤心”“愤怒”“吃惊”“跑步”“走路”“打招呼”等别名，以及“浅色皮肤”“小麦色皮肤”“深色皮肤”。不支持的修饰词不会新增画面特征；“可爱”“写实”等词不是可控风格参数。

手指属于简化卡通画法，不是写实手部建模。各人物类型的主要比例区别是儿童与成人的头身比例；发型、衣服和体型可独立选择。任意职业服装、装备和自动复杂场景生成仍不在当前范围内；手工多角色构图与连续动作使用上述动画入口。

## 5. 单独查看生成指令

单次推理，只在终端输出指令：

```powershell
python infer.py "棕色熊坐着"
```

交互式推理，可连续输入提示词，输入空行退出：

```powershell
python infer.py
```

输出使用 `|` 分隔多条命令。渲染器同时支持 `|` 分隔和换行分隔。不要把交互提示文字一并保存到指令文件。

## 6. 从指令文件渲染

将绘图指令保存为 UTF-8 编码的 `commands.txt`，例如：

```text
canvas 64 64 transparent
ellipse 32 32 12 10 #e8923f
ellipse 28 29 1.5 2 #ffffff
ellipse 36 29 1.5 2 #ffffff
dot 28 29 #292632 0.8
dot 36 29 #292632 0.8
line 29 36 35 36 #292632 1
```

执行：

```powershell
python genimg.py --instructions commands.txt --output custom.png --size 1024
```

支持的指令格式：

```text
canvas 宽 高 背景色
ellipse 中心X 中心Y 半径X 半径Y 颜色
rect 左上X 左上Y 右下X 右下Y 颜色
tri X1 Y1 X2 Y2 X3 Y3 颜色
line X1 Y1 X2 Y2 颜色 [线宽]
dot X Y 颜色 [半径]
curve 起点X 起点Y 控制点1X 控制点1Y 控制点2X 控制点2Y 终点X 终点Y 起点半径 终点半径 颜色
```

坐标和线宽使用原始画布单位，不是最终 PNG 像素。颜色建议使用 `#RRGGBB`，画布背景可使用 `transparent`。命令按输入顺序叠加，后面的图形覆盖前面的图形。

`curve` 的两个半径必须大于 0，表示轮廓半径而非直径；粗细沿曲线渐变。例如下面是一条弯曲、渐细的尾巴：

```text
curve 40 33 48 38 55 28 50 22 2.4 0.6 #e8923f
```

## 7. 参数说明

### 渲染参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--instructions` | 无 | UTF-8 指令文件路径；未指定时读取标准输入 |
| `--output` | `output.png` | 输出文件路径，建议使用 `.png` 后缀；父目录须已存在 |
| `--size` | `1024` | 正方形输出边长，范围 64 到 4096 |
| `--outline-width` | `0.8` | 外轮廓描边宽度，使用画布单位；设为 0 关闭 |
| `--background` | 无 | 合成背景色，如 `"#ffffff"` |
| `--fixed-canvas` | 关闭 | 保留逻辑坐标和固定画布，不自动紧贴透明内容；逐帧渲染应使用此参数 |

当前透明模板无需指定 `--background`。自定义指令如需透明背景，请将 `canvas` 的背景设置为 `transparent`；`--background transparent` 不会移除指令中已经绘制的不透明背景。

### 推理参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| 位置参数 | 无 | 中文提示词；未指定时进入交互模式 |
| `--max-new` | 自动 | 默认生成到模型剩余上下文上限；指定时必须大于 0 |
| `--template` | 关闭 | 直接生成新版模板指令，不加载模型，适合重训前检查造型 |

参数帮助：

```powershell
python infer.py --help
python genimg.py --help
```

## 8. 常见问题

### 提示需要通过文件或标准输入提供绘图指令

说明仅运行了渲染器，没有提供输入。使用完整管道命令：

```powershell
python infer.py "棕色熊坐着" | python genimg.py --output bear.png
```

### 找不到模型或提示模型格式过旧

检查 `v2/tiny_model.pt` 是否存在。旧字符级模型与当前 `dsl-token-v1` tokenizer 不兼容，需要按第 3 节重新生成数据并训练。不要将根目录的旧模型直接复制到 `v2` 使用。

### 当前模型词表不支持这个提示词

即使旧模型使用同一种 tokenizer，也可能没有人物或新元素的词表。先加 `--template` 预览；准备好后使用最新 `data.txt` 重新训练。旧模型不具备新增选项的学习能力。

### 无法识别动物或动作

动物请使用第 4 节支持的名称，例如 `棕色熊坐着`，不要省略动作。人物可以省略动作，默认站着；环境元素不需要动作。多主体或多动作应拆成多次生成。

### 没有找到可绘制指令或提示跳过无效指令

先单独运行 `python infer.py "棕色熊坐着"`，检查输出或前面的报错。生成指令应包含 `canvas` 和绘图命令。渲染器会跳过部分参数错误的命令，成功保存图片不代表所有指令都有效。

### 中文提示词乱码

先在同一 PowerShell 终端设置 UTF-8，再重新执行命令：

```powershell
chcp 65001
$env:PYTHONUTF8 = "1"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
```

### 成功出图是否代表模型训练成功

不一定。当前推理器在没有生成结束符或指令语法无效时，会回退到对应模板，并在标准错误输出中明确提示。短元素不再因为命令数量少而被强制回退。校验覆盖命令种类、参数数量、数值和颜色合法性，但不保证模型语义正确，也不能保证它遵从了全部人物选项。

## 9. 游戏和动画使用边界

输出是静态 RGBA PNG，可作为简单 2D 角色原型或素材草稿，但不能仅凭分辨率认定达到生产质量。扩大 `--size` 会改善边缘输出，不会增加角色设计细节。

当前不输出 SVG、骨骼、分层文件、连续动画帧或精灵图集。不同动作的透明图片会独立裁剪和居中，不能直接保证动画帧之间的角色比例、位置和锚点一致。用于游戏或动画前，需人工检查造型、颜色、边缘，并统一帧画布、缩放和锚点。

要扩展题材或显著提高美术质量，应先改进角色模板或引入质量更高且有使用授权的数据，再重新训练。单纯增加训练轮数无法突破模板本身的视觉质量上限。

## 10. 回归检查

在 `v2` 目录中执行：

```powershell
python -m unittest test_genimg -v
```

检查曲线弯曲和渐细、旧指令圆头兼容、全部 2,870 条样本的解析和渲染、人物选项的实际像素变化、月牙透明性，以及长人物与短元素的解码路径。解码测试使用记录好的 token 输出，不评估实际训练模型的美术质量。