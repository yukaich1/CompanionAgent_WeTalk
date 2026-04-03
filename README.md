# Wetalk

Wetalk 是一个基于 Python 与 Flask 的智能角色扮演与虚拟陪伴系统。

当前版本已经按新的分层方案完成重构，核心目标是：

- 保留角色聊天、人设学习、工具调用、记忆、情绪与关系等现有能力
- 将原本集中在 `main.py` 的松散流程拆分为清晰的模块
- 让“人设 / 记忆 / 路由 / 上下文编排 / 工具 / 推理 / 诊断”具备明确边界

## 当前能力

- Web 聊天界面
- 角色头像上传与裁切
- 角色资料上传、预览、确认写入
- 本地资料优先的人设 RAG
- 联网搜索与天气工具
- 记忆、关系、情绪、思考链协同
- 新架构状态持久化

## 项目结构

```text
Ireina/
├── app.py
├── main.py
├── config.py
├── const.py
├── llm.py
├── tools/
│   ├── base.py
│   ├── registry.py
│   ├── runtime.py
│   ├── web_search.py
│   └── weather.py
├── knowledge/
│   ├── knowledge_source.py
│   ├── persona_system.py
│   ├── persona_rag_engine.py
│   ├── persona_conflict_filter.py
│   ├── persona_evolution_engine.py
│   └── vault_version_manager.py
├── memory/
│   ├── memory_system.py
│   ├── memory_rag_engine.py
│   └── memory_writer.py
├── routing/
│   ├── query_router.py
│   └── query_rewriter.py
├── context/
│   ├── context_assembler.py
│   └── recall_deduplicator.py
├── reasoning/
│   ├── emotion_state_machine.py
│   └── thought_system.py
├── diagnostics/
│   ├── self_check.py
│   ├── conflict_log.py
│   └── health_monitor.py
├── templates/
├── static/
└── uploads/
```

### 模块说明

- `knowledge/`：角色本体、人设状态、人设证据库与人设召回
- `memory/`：记忆状态、记忆召回、记忆摘要写回
- `routing/`：问题分类、搜索改写
- `context/`：上下文组装、召回去重、预算分配
- `tools/`：网页搜索、天气查询等工具
- `reasoning/`：情绪状态机与思考系统
- `diagnostics/`：自检、冲突记录、健康监控

## 环境要求

- Windows
- Python 3.13
- 可用的 LLM API Key

## 安装依赖

```powershell
cd C:\Users\0yoyx\Desktop\code\python\Ireina
python -m pip install -r requirements.txt
python -m pip install faiss-cpu
```

如果你使用虚拟环境：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install faiss-cpu
```

## LLM 配置

项目支持通用 LLM 配置，不再写死某一家接口。

在项目根目录创建 `.env`。

### Mistral 示例

```env
LLM_PROVIDER="mistral"
LLM_API_KEY="你的key"
LLM_CHAT_MODEL="mistral-medium-latest"
LLM_EMBEDDING_MODEL="mistral-embed"
```

### OpenAI 示例

```env
LLM_PROVIDER="openai"
LLM_API_KEY="你的key"
LLM_CHAT_MODEL="gpt-4.1-mini"
LLM_EMBEDDING_MODEL="text-embedding-3-small"
```

### 自定义兼容接口示例

```env
LLM_PROVIDER="openai_compatible"
LLM_API_KEY="你的key"
LLM_BASE_URL="https://你的接口地址/v1"
LLM_CHAT_MODEL="你的聊天模型名"
LLM_EMBEDDING_MODEL="你的嵌入模型名"
```

常用字段：

- `LLM_PROVIDER`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_CHAT_MODEL`
- `LLM_EMBEDDING_MODEL`

## 启动项目

```powershell
python app.py
```

然后打开：

[http://127.0.0.1:5000](http://127.0.0.1:5000)

## 主要使用流程

### 1. 学习角色资料

- 上传文本文件或直接粘贴角色资料
- 系统自动生成待确认预览
- 确认写入后，人设进入本地人设库
- 回答角色设定相关问题时，优先从本地资料中检索

### 2. 工具使用

当前工具位于 `tools/`：

- `web_search`
- `weather`

工作原则：

- 角色强相关问题：先查本地人设资料，再补强相关搜索
- 现实信息问题：优先使用工具
- 没有依据时：保守回答，避免编造

### 3. 记忆与关系

系统会维护：

- 角色与用户的关系状态
- 情绪状态机
- 近期活动
- 记忆写回与召回结果

## 运行时数据文件

常见运行文件：

- `memory_state.json`
- `persona_state.json`
- `frontend_state.json`
- `uploads/`
- `ireina_save.pkl`
- `ireina_persona.pkl`

其中：

- `memory_state.json`、`persona_state.json` 是当前新架构状态文件
- `ireina_save.pkl`、`ireina_persona.pkl` 主要用于旧数据兼容与迁移

## GitHub 上传建议

推荐上传：

- 源代码
- `README.md`
- `requirements.txt`
- `.gitignore`
- `templates/`
- `static/`

不要上传：

- `.env`
- `uploads/`
- `frontend_state.json`
- `memory_state.json`
- `persona_state.json`
- `ireina_save.pkl`
- `ireina_persona.pkl`
- `.venv/`
- `.idea/`

## 调试命令

检查导入：

```powershell
python -c "import app; import main; print('ok')"
```

检查语法：

```powershell
python -m py_compile app.py main.py
```

## 当前重构状态

当前版本已经完成以下替换：

- 根目录旧 `persona_system.py` 已移除
- 根目录旧 `memory_system.py` 已移除
- 根目录旧 `belief_system.py` 已移除
- 根目录旧 `emotion_system.py` 已移除
- 根目录旧 `thought_system.py` 已移除

主链现在统一走：

- `knowledge/persona_system.py`
- `memory/memory_system.py`
- `routing/query_router.py`
- `routing/query_rewriter.py`
- `context/context_assembler.py`
- `context/recall_deduplicator.py`
- `reasoning/emotion_state_machine.py`
- `reasoning/thought_system.py`
- `tools/runtime.py`

`main.py` 负责流程编排，不再作为主要业务逻辑堆积点。
