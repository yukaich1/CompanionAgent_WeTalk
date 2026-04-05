# Wetalk

一个面向情感陪伴与角色扮演场景的 AI Agent 项目。

Wetalk 的目标不是做一个“普通聊天机器人”，而是构建一个能够：
- 学习角色资料并形成稳定的人设基础模板
- 在多轮对话中保持较强的人格一致性与情感连续性
- 结合人设知识库、记忆系统与外部工具进行 grounded response
- 在需要时调用天气、联网搜索等工具

## 项目特点

- 角色基础模板系统
  使用“分析师”Prompt 将原始材料提炼为结构化角色基础模板，再在每轮对话中通过“演员”Prompt 注入模型。
- 人设 RAG
  将角色故事、经历、风格示例、关键词等内容分块存入知识库，用于后续精准检索与动态注入。
- 双层记忆
  同时维护结构化关系状态与多轮对话记忆，支持更自然的情绪延续和关系变化。
- 情绪与慢思考
  使用情绪状态机与 Thought System 对回复策略、语气、工具依赖进行中间层规划。
- 工具路由
  支持天气查询与联网搜索，并通过意图提取与查询改写将工具结果结构化注入回复。
- Web 界面
  提供角色资料上传、头像裁切、预览确认、流式多气泡回复、调试证据区等前端能力。

## 当前架构

```text
Ireina/
├── app.py                         # Flask Web 入口
├── main.py                        # 主编排器 AISystem
├── const.py                       # 核心 Prompt 与常量
├── llm.py                         # 模型与 embedding 调用层
├── ai_runtime_support.py          # 运行期 prompt 注入、调试与辅助工具
├── persona_prompting.py           # 分析师 / 演员 Prompt 构建
├── persona_models.py              # 人设预览与模板模型
│
├── knowledge/
│   ├── persona_system.py          # 人设门面层
│   ├── persona_ingest_service.py  # 人设学习与入库
│   ├── persona_preview_service.py # 预览生成与联网补充
│   ├── persona_context_service.py # 人设检索与上下文构建
│   ├── persona_rag_engine.py      # 人设 RAG 引擎
│   ├── persona_state.py           # 人设状态模型
│   ├── persona_shared.py          # 共享 schema / 常量
│   └── ...
│
├── memory/
│   ├── memory_system.py           # 记忆系统状态与存储
│   ├── memory_rag_engine.py       # 记忆检索
│   ├── memory_writer.py           # 对话写回
│   └── ...
│
├── reasoning/
│   ├── thought_system.py          # 慢思考链
│   └── emotion_state_machine.py   # 情绪与关系状态机
│
├── routing/
│   ├── query_router.py            # 问题路由
│   └── query_rewriter.py          # 查询改写
│
├── tools/
│   ├── intent_extractor.py        # 轻量意图提取
│   ├── tool_router.py             # 工具调度与降级
│   ├── runtime.py                 # 工具执行运行时
│   ├── weather.py                 # 天气工具
│   └── web_search.py              # 联网搜索工具
│
├── context/
│   ├── context_assembler.py       # 上下文拼装
│   └── recall_deduplicator.py     # 去重
│
├── static/
├── templates/
└── uploads/
```

## 运行环境

推荐环境：
- Python 3.11+
- Windows / macOS / Linux
- 一个可用的 LLM API Key

## 安装

在项目根目录执行：

```powershell
cd 你的项目文件夹
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果你使用的是 macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置 .env

在项目根目录创建 `.env` 文件。

### 使用 Mistral

```env
LLM_PROVIDER="mistral"
LLM_API_KEY="your_api_key"
LLM_CHAT_MODEL="mistral-medium-latest"
LLM_EMBEDDING_MODEL="mistral-embed"
```

### 使用 OpenAI

```env
LLM_PROVIDER="openai"
LLM_API_KEY="your_api_key"
LLM_CHAT_MODEL="gpt-4.1-mini"
LLM_EMBEDDING_MODEL="text-embedding-3-small"
```

### 使用 OpenAI 兼容接口

```env
LLM_PROVIDER="openai_compatible"
LLM_API_KEY="your_api_key"
LLM_BASE_URL="https://your-endpoint/v1"
LLM_CHAT_MODEL="your-chat-model"
LLM_EMBEDDING_MODEL="your-embedding-model"
```

常用字段说明：
- `LLM_PROVIDER`：模型提供方
- `LLM_API_KEY`：API Key
- `LLM_BASE_URL`：兼容接口地址，可选
- `LLM_CHAT_MODEL`：聊天模型名
- `LLM_EMBEDDING_MODEL`：向量模型名

## 启动项目

```powershell
cd 你的项目路径
python app.py
```

然后在浏览器打开：

[http://127.0.0.1:5000](http://127.0.0.1:5000)

## 使用流程

1. 打开 Web 页面。
2. 在“角色学习”区域上传资料文件，或直接粘贴角色设定文本。
3. 生成待确认预览。
4. 查看：
   - 角色基础模板
   - 候选关键词
   - 本地资料与联网补充摘要
5. 选择最多 8 个高权重关键词并确认写入。
6. 开始对话。

## 人设学习逻辑

项目当前采用两阶段人设学习：

1. 分析阶段
   - 使用 `build_base_template_generation_prompt()`
   - 由“专业角色分析师”Prompt 将材料提炼为结构化基础模板
   - 输出包含：
     - `base_template`
     - `character_voice_card`
     - `display_keywords`
     - `style_examples`
     - `story_chunks`

2. 扮演阶段
   - 使用 `build_base_template_injection_prompt()`
   - 由“专业演员”Prompt 将角色基础模板转成每轮对话时的扮演约束

## 工具调用逻辑

项目中，工具不是通过简单关键词硬匹配触发，而是采用：

1. `IntentExtractor` 提取意图
2. `QueryRouter` 映射路由
3. `QueryRewriter` 生成更适合检索 / 搜索的 query
4. `ToolRouter` 执行天气 / 搜索工具
5. 工具结果以结构化方式注入最终回复 Prompt

当前支持：
- 天气查询
- 网络搜索

![聊天界面演示](./assets/141804.png)

## 注意事项

- 如果你修改了 Prompt 或知识库结构，建议重新生成角色预览后再确认写入。
- 如果前端界面没有立即刷新最新状态，可以清空浏览器缓存后重试。


## 后续建议

如果你接下来继续迭代，建议优先关注：
- 角色模板生成质量
- 故事块检索命中率
- 工具结果与角色语气的自然融合
- 开发与丰富更多工具
