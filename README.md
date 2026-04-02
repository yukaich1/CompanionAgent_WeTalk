# Wetalk

Wetalk 是一个基于 Python、Flask 和大语言模型的智能角色扮演与虚拟陪伴系统。

它的核心目标是：

- 让角色能够学习你提供的人设资料
- 在聊天中尽量维持人物性格、说话方式、价值观和经历一致性
- 结合本地资料、记忆与联网信息进行更可靠的回复
- 通过 Web 界面完成聊天、角色配置、头像编辑与人设学习


## 功能概览

- Web 聊天界面
- 多气泡角色回复
- 头像上传与裁切
- 角色名称修改
- 人设资料上传与文本学习
- 自动生成人设预览并确认写入
- 本地人设检索
- 工具化联网搜索
- 天气查询
- 记忆系统
- 情绪与好感度系统
- 思考与信念系统
- 最近活动记录


## 项目结构

主要文件如下：

- [app.py](C:\Users\0yoyx\Desktop\code\python\Ireina\app.py)
  Web 服务入口，负责页面与 API。

- [main.py](C:\Users\0yoyx\Desktop\code\python\Ireina\main.py)
  AI 主调度系统，负责把人设、记忆、情绪、思考、工具串起来。

- [llm.py](C:\Users\0yoyx\Desktop\code\python\Ireina\llm.py)
  通用 LLM 接口层，支持通过统一配置切换不同供应商。

- [persona_system.py](C:\Users\0yoyx\Desktop\code\python\Ireina\persona_system.py)
  人设学习、提炼、检索与持久化。

- [persona_prompting.py](C:\Users\0yoyx\Desktop\code\python\Ireina\persona_prompting.py)
  人设提取相关 prompt。

- [persona_models.py](C:\Users\0yoyx\Desktop\code\python\Ireina\persona_models.py)
  人设结构化数据模型。

- [memory_system.py](C:\Users\0yoyx\Desktop\code\python\Ireina\memory_system.py)
  记忆与混合检索。

- [emotion_system.py](C:\Users\0yoyx\Desktop\code\python\Ireina\emotion_system.py)
  情绪、关系与好感度逻辑。

- [thought_system.py](C:\Users\0yoyx\Desktop\code\python\Ireina\thought_system.py)
  后台思考与反思逻辑。

- [belief_system.py](C:\Users\0yoyx\Desktop\code\python\Ireina\belief_system.py)
  角色信念系统。

- [tools](C:\Users\0yoyx\Desktop\code\python\Ireina\tools)
  Agent tools 层，目前包含 `web_search` 与 `weather`。

- [templates](C:\Users\0yoyx\Desktop\code\python\Ireina\templates)
  Flask 模板。

- [static](C:\Users\0yoyx\Desktop\code\python\Ireina\static)
  前端脚本、样式与静态资源。


## 环境要求

建议环境：

- Windows
- Python 3.13
- 可用的 LLM API Key


## 安装依赖

先进入项目目录：

```powershell
cd C:\Users\0yoyx\Desktop\code\python\Ireina
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install faiss-cpu
```

如果你使用虚拟环境，也可以改成：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install faiss-cpu
```


## 通用 LLM 配置

Wetalk 现在不再写死只能用 Mistral。

你可以通过统一配置接入不同供应商，只需要在 [`.env`](C:\Users\0yoyx\Desktop\code\python\Ireina\.env) 里填写：

- `LLM_PROVIDER`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_CHAT_MODEL`
- `LLM_EMBEDDING_MODEL`

其中：

- `LLM_PROVIDER` 用来选择供应商预设
- `LLM_API_KEY` 是该供应商的 key
- `LLM_BASE_URL` 可以覆盖默认地址
- `LLM_CHAT_MODEL` 是聊天模型
- `LLM_EMBEDDING_MODEL` 是 embedding 模型


## 推荐配置方式

### 1. Mistral

```env
LLM_PROVIDER="mistral"
LLM_API_KEY="你的_mistral_api_key"
LLM_CHAT_MODEL="mistral-medium-latest"
LLM_EMBEDDING_MODEL="mistral-embed"
```


### 2. OpenAI

```env
LLM_PROVIDER="openai"
LLM_API_KEY="你的_openai_api_key"
LLM_CHAT_MODEL="gpt-4.1-mini"
LLM_EMBEDDING_MODEL="text-embedding-3-small"
```


### 3. 自定义 OpenAI 兼容接口

```env
LLM_PROVIDER="openai_compatible"
LLM_API_KEY="你的_api_key"
LLM_BASE_URL="https://你的接口地址/v1"
LLM_CHAT_MODEL="你的聊天模型名"
LLM_EMBEDDING_MODEL="你的向量模型名"
```


## 当前内置 provider 预设

当前 `llm.py` 中内置了这些 provider 名称：

- `mistral`
- `openai`
- `openrouter`
- `siliconflow`
- `deepseek`
- `custom`
- `openai_compatible`

说明：

- 对于没有默认 embedding 模型的提供方，你需要自己填写 `LLM_EMBEDDING_MODEL`
- 如果提供方本身不支持 embedding，那么记忆和人设向量检索能力会受影响


## 兼容旧配置

为了兼容旧版本，当前仍然支持这些旧环境变量：

- `MISTRAL_API_KEY`
- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `OPENROUTER_API_KEY`
- `SILICONFLOW_API_KEY`

但更推荐统一使用：

```env
LLM_PROVIDER=...
LLM_API_KEY=...
LLM_CHAT_MODEL=...
LLM_EMBEDDING_MODEL=...
```


## 启动项目

启动 Web 服务：

```powershell
python app.py
```

然后打开浏览器访问：

[http://127.0.0.1:5000](http://127.0.0.1:5000)


## Web 界面说明

主界面主要包含三块：

- 左侧角色状态
  显示角色名、心情、好感度、人设关键词、最近活动。

- 中间聊天区
  与角色对话，回复支持多气泡展示，并带更自然的逐步出现效果。

- 右上角配置面板
  可以修改角色名、上传头像、上传人设资料、清空人设、删除存档。


## 头像功能

支持上传头像图片：

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`

上传后会进入头像编辑器，你可以：

- 拖拽图片位置
- 通过滑杆缩放
- 选定最终头像区域


## 人设学习方式

Wetalk 当前的人设学习以“用户资料优先”为原则。

你可以通过两种方式提供资料：

1. 上传文件
2. 直接粘贴文本

支持的资料格式：

- `.txt`
- `.md`
- `.log`
- `.json`
- `.csv`

学习流程如下：

1. 用户提供资料
2. 系统解析本地资料
3. 系统自动补充联网参考信息
4. 生成人设预览
5. 你确认后正式写入人设库

注意：

- 联网信息是补充，不会覆盖你的原始资料
- 预览中会保留你提供资料的内容
- 确认写入后，预览会自动清空


## 人设学习重点

系统重点学习这些内容：

- 性格特质
- 说话方式
- 价值观
- 世界观
- 喜好与厌恶
- 角色特点
- 口头禅
- 称呼习惯
- 句末习惯
- 可自然提及的经历

目标是：

- 角色特质作为“长期底色”
- 不机械重复人设标签
- 不强行每一句都表演性格
- 对故事、设定、经历类问题优先依据资料与检索结果回答
- 尽量减少编造


## Tools 结构

当前项目已经使用 `tools` 架构，而不是简单工具模块。

现有工具位于 [tools](C:\Users\0yoyx\Desktop\code\python\Ireina\tools)：

- [tools\web_search.py](C:\Users\0yoyx\Desktop\code\python\Ireina\tools\web_search.py)
  用于角色设定、背景、故事等参考信息检索。

- [tools\weather.py](C:\Users\0yoyx\Desktop\code\python\Ireina\tools\weather.py)
  用于实时天气查询。

- [tools\runtime.py](C:\Users\0yoyx\Desktop\code\python\Ireina\tools\runtime.py)
  负责工具调度与执行。

- [tools\registry.py](C:\Users\0yoyx\Desktop\code\python\Ireina\tools\registry.py)
  负责工具注册。

后续如果要继续扩展工具，建议直接往 `tools` 层新增。


## 数据存储

运行后会在项目目录生成或使用这些文件：

- [ireina_save.pkl](C:\Users\0yoyx\Desktop\code\python\Ireina\ireina_save.pkl)
  AI 主状态存档。

- [ireina_persona.pkl](C:\Users\0yoyx\Desktop\code\python\Ireina\ireina_persona.pkl)
  人设库与人设索引数据。

- [frontend_state.json](C:\Users\0yoyx\Desktop\code\python\Ireina\frontend_state.json)
  前端状态，如最近活动与头像路径。

- [uploads](C:\Users\0yoyx\Desktop\code\python\Ireina\uploads)
  上传文件目录。


## 删除存档会做什么

在前端点“删除全部存档”后，会清空：

- 聊天记录
- 人设库
- 角色相关持久化状态
- 前端记录的头像状态

删除后不会自动生成开场对白。


## 常见使用流程

### 1. 启动项目

```powershell
python app.py
```

### 2. 配置 LLM

在 `.env` 中填写 provider 和 key，例如：

```env
LLM_PROVIDER="mistral"
LLM_API_KEY="你的key"
LLM_CHAT_MODEL="mistral-medium-latest"
LLM_EMBEDDING_MODEL="mistral-embed"
```

### 3. 上传角色资料

- 打开右上角配置
- 上传文件或粘贴文本
- 查看自动生成的人设预览
- 点击确认写入

### 4. 开始聊天

你可以直接提问：

- 日常聊天
- 角色设定
- 角色经历
- 喜欢什么
- 讨厌什么
- 口头禅是什么

对于强相关人设问题，系统会优先：

1. 检索用户资料
2. 检索本地人设库
3. 联网补充强相关信息

如果证据不足，应尽量保守回答，而不是编造。


## 常见问题

### 1. 为什么人设学习有时较慢

因为当前流程包含：

- 文本解析
- 结构化提炼
- embedding
- 向量入库
- 联网补充

资料越大，学习时间越长。


### 2. 为什么有时联网结果不足

联网检索依赖当前可访问的公开来源与模型服务状态。
如果外部资料不充分，系统会优先使用你上传的资料，而不是乱编。


### 3. 如果上传了重复资料怎么办

系统会对重复资料做指纹检测，并尽量跳过重复内容，不会无限重复入库。


### 4. 如果更换 LLM 供应商会怎样

只要新供应商：

- 提供兼容的聊天接口
- 提供可用的 embedding 接口

就可以继续保留 Wetalk 的大部分能力。

如果只有聊天没有 embedding，那么：

- 聊天可能还能用
- 但人设向量检索、记忆向量检索会受影响


## 开发说明

如果你要继续开发，建议优先阅读：

- [llm.py](C:\Users\0yoyx\Desktop\code\python\Ireina\llm.py)
- [main.py](C:\Users\0yoyx\Desktop\code\python\Ireina\main.py)
- [persona_system.py](C:\Users\0yoyx\Desktop\code\python\Ireina\persona_system.py)
- [emotion_system.py](C:\Users\0yoyx\Desktop\code\python\Ireina\emotion_system.py)
- [memory_system.py](C:\Users\0yoyx\Desktop\code\python\Ireina\memory_system.py)
- [app.py](C:\Users\0yoyx\Desktop\code\python\Ireina\app.py)
- [static\app.js](C:\Users\0yoyx\Desktop\code\python\Ireina\static\app.js)


## 当前状态

当前 Wetalk 已完成从桌面窗口版向 Web 产品版的迁移，并保留了：

- 聊天
- 人设学习
- 情绪与好感度
- 记忆
- 工具调用

后续仍适合继续优化：

- 更稳定的后端流式输出
- 更成熟的工具调度
- 更严格的人设取证
- 更多可扩展的 tools
