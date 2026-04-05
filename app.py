from __future__ import annotations

import copy
import json
import shutil
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

from const import APP_DIR, NEW_MEMORY_STATE_PATH, NEW_PERSONA_STATE_PATH, SAVE_PATH
from main import AIConfig, AISystem, PersonalityConfig, check_has_valid_key


BASE_DIR = Path(APP_DIR)
STATE_PATH = BASE_DIR / "frontend_state.json"
UPLOAD_DIR = BASE_DIR / "uploads"
AVATAR_DIR = UPLOAD_DIR / "avatars"
ALLOWED_TEXT_SUFFIXES = {".txt", ".md", ".log", ".json", ".csv", ".pdf"}
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_DIR.mkdir(parents=True, exist_ok=True)


def _default_frontend_state() -> dict:
    return {
        "avatar_path": "",
        "recent_activity": [],
        "persona_web_search_enabled": True,
    }


def _load_frontend_state() -> dict:
    if not STATE_PATH.exists():
        return _default_frontend_state()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_frontend_state()
    state = _default_frontend_state()
    if isinstance(data, dict):
        state.update(data)
    if not isinstance(state.get("recent_activity"), list):
        state["recent_activity"] = []
    state["persona_web_search_enabled"] = bool(state.get("persona_web_search_enabled", True))
    return state


def _save_frontend_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_activity(state: dict, text: str) -> None:
    activity = state.setdefault("recent_activity", [])
    activity.insert(0, {"text": text, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
    del activity[10:]
    _save_frontend_state(state)


def _ensure_avatar_square(file_path: Path) -> None:
    with Image.open(file_path) as image:
        image = image.convert("RGB")
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((512, 512))
        image.save(file_path, format="PNG")


def _read_persona_text(file_path: Path, suffix: str) -> str:
    suffix = (suffix or "").lower()
    if suffix == ".pdf":
        if PdfReader is None:
            raise ValueError("当前环境未安装 PDF 解析依赖，请先安装 pypdf，或改为上传 txt / md / json 文件。")
        try:
            reader = PdfReader(str(file_path))
            text = "\n".join((page.extract_text() or "").strip() for page in reader.pages)
        except Exception as exc:
            raise ValueError(f"PDF 读取失败：{exc}") from exc
        text = text.strip()
        if not text:
            raise ValueError("这个 PDF 没有解析出可用文本，请更换为文本版资料或可复制文本的 PDF。")
        return text

    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            text = file_path.read_text(encoding=encoding).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("文件没有解析出可用文本，请确认不是空文件，或改用 txt / md / json / pdf。")
    return text


def _split_bubbles(text: str) -> list[str]:
    import re

    normalized = (text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return []
    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if len(lines) > 1:
        return lines
    if len(normalized) <= 90:
        return [normalized]

    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*", normalized) if part.strip()]
    if len(sentences) <= 1:
        return [normalized]

    target = 90 if len(normalized) < 260 else 120
    bubbles: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) > target:
            bubbles.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)
    if current:
        bubbles.append(" ".join(current).strip())
    return [bubble for bubble in bubbles if bubble]


def _friendliness_hearts(value: float) -> str:
    normalized = max(0.0, min(1.0, float(value)))
    heart_count = max(0, min(5, int(round(normalized * 5))))
    return "❤" * heart_count if heart_count else "♡"


def _mood_to_zh(mood_name: str) -> str:
    mapping = {
        "neutral": "平静",
        "exuberant": "雀跃",
        "dependent": "依恋",
        "relaxed": "放松",
        "docile": "温顺",
        "bored": "低落",
        "anxious": "紧张",
        "disdainful": "冷淡",
        "hostile": "不悦",
        "calm": "平静",
        "happy": "愉快",
        "concerned": "关切",
        "hurt": "受伤",
        "平静": "平静",
        "愉快": "愉快",
        "关切": "关切",
        "受伤": "受伤",
    }
    return mapping.get(mood_name, mood_name)


def _serialize_message(message: dict) -> dict:
    content = message.get("content")
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        content = "\n".join(part for part in parts if part).strip()
    if not isinstance(content, str):
        content = str(content or "")
    role = message.get("role", "assistant")
    bubbles = _split_bubbles(content) if role == "assistant" else ([content.strip()] if content.strip() else [])
    return {"role": role, "content": content, "bubbles": bubbles}


def _bootstrap_ai() -> AISystem:
    ai = AISystem.load(SAVE_PATH)
    if ai is None:
        ai = AISystem()
    ai.on_startup()
    return ai


def _new_relation_snapshot():
    state = getattr(_ai_system, "new_memory_state", None)
    return getattr(state, "relation_state", None) if state is not None else None


def _current_mood_snapshot():
    machine = getattr(_ai_system, "emotion_state_machine", None)
    return getattr(machine, "current_state", None) if machine is not None else None


def _new_persona_snapshot():
    return getattr(_ai_system, "new_persona_state", None)


def _save_ai() -> None:
    _ai_system.save(SAVE_PATH)


def _clear_uploaded_files() -> None:
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)


def _set_agent_name(new_name: str) -> None:
    personality = _ai_system.config.personality
    current_name = _ai_system.config.name
    system_prompt = _ai_system.config.system_prompt.replace(current_name, new_name) if current_name else _ai_system.config.system_prompt
    new_config = AIConfig(
        name=new_name,
        system_prompt=system_prompt,
        personality=PersonalityConfig(
            open=personality.open,
            conscientious=personality.conscientious,
            agreeable=personality.agreeable,
            extrovert=personality.extrovert,
            neurotic=personality.neurotic,
        ),
    )
    _ai_system.set_config(new_config)
    pending_previews = getattr(_ai_system.persona_system, "pending_previews", {}) or {}
    for preview in pending_previews.values():
        if isinstance(preview, dict):
            preview_name = str(preview.get("persona_name", "") or "").strip()
            if not preview_name or preview_name == current_name:
                preview["persona_name"] = new_name


def _build_snapshot() -> dict:
    history = []
    for message in _ai_system.get_message_history(False):
        if message.get("role") == "system":
            continue
        serialized = _serialize_message(message)
        if serialized["content"]:
            history.append(serialized)

    avatar_path = _frontend_state.get("avatar_path", "")
    avatar_url = ""
    if avatar_path:
        try:
            avatar_url = "/uploads/" + Path(avatar_path).relative_to(UPLOAD_DIR).as_posix()
        except Exception:
            avatar_url = ""

    relation_state = _new_relation_snapshot()
    emotion_state = _current_mood_snapshot()
    persona_state = _new_persona_snapshot()
    persona_identity = getattr(getattr(persona_state, "immutable_core", None), "identity", None)
    persona_metadata = getattr(persona_state, "metadata", {}) if persona_state is not None else {}
    persona_keywords = list(persona_metadata.get("display_keywords", []) or [])
    persona_chunk_count = len(getattr(getattr(persona_state, "evidence_vault", None), "parent_chunks", []) or [])
    if not persona_chunk_count:
        persona_chunk_count = _ai_system.persona_system.chunk_count

    has_user_conversation = any(message.get("role") == "user" for message in _ai_system.get_message_history(False))
    if has_user_conversation and emotion_state is not None:
        mood_name = _mood_to_zh(emotion_state.mood)
        mood_description = f"当前情绪：{mood_name}，强度 {round(float(emotion_state.intensity), 2)}。"
    elif has_user_conversation:
        mood_name = _mood_to_zh(_ai_system.emotion_system.get_mood_name())
        mood_description = _ai_system.emotion_system.get_mood_prompt()
    else:
        mood_name = "平静"
        mood_description = "情绪稳定，尚未因对话产生明显波动。"

    if has_user_conversation and relation_state is not None:
        friendliness = round(float(relation_state.affection), 2)
        affinity = _friendliness_hearts(relation_state.affection)
    elif has_user_conversation:
        friendliness = round((_ai_system.relation_system.friendliness + 100) / 200, 2)
        affinity = _friendliness_hearts((_ai_system.relation_system.friendliness + 100) / 200)
    else:
        friendliness = 0.0
        affinity = "♡"

    return {
        "agent": {
            "name": getattr(persona_identity, "name", "") or _ai_system.config.name,
            "avatarUrl": avatar_url,
            "mood": mood_name,
            "moodDescription": mood_description,
            "friendliness": friendliness,
            "affinity": affinity,
            "personaChunks": persona_chunk_count,
            "keywords": persona_keywords or _ai_system.persona_system.get_display_keywords(),
            "personaStatus": _ai_system.persona_system.get_status().dict(),
        },
        "settings": {
            "personaWebSearchEnabled": bool(_frontend_state.get("persona_web_search_enabled", True)),
        },
        "debug": getattr(_ai_system, "last_debug_info", {}) or {},
        "history": history,
        "recentActivity": _frontend_state.get("recent_activity", []),
    }


def _json_ok(**payload):
    return jsonify({"ok": True, **payload})


def _json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


app = Flask(__name__, template_folder="templates", static_folder="static")
_ai_lock = Lock()
_frontend_state = _load_frontend_state()
_ai_system = _bootstrap_ai() if check_has_valid_key() else AISystem()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/api/bootstrap")
def api_bootstrap():
    with _ai_lock:
        return jsonify(_build_snapshot())


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(force=True, silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return _json_error("消息不能为空。")

    with _ai_lock:
        backup = copy.deepcopy(_ai_system)
        try:
            response = _ai_system.send_message(message)
            _save_ai()
            _append_activity(_frontend_state, f"{_ai_system.config.name} 完成了一轮对话")
        except Exception as exc:
            traceback.print_exc()
            fallback_reply = "刚刚这条消息我没能顺利接住，可能是网络或模型服务暂时不稳定。你可以再发一次，我会继续。"
            try:
                history = _ai_system.get_message_history(False)
                if history and history[-1].get("role") != "assistant":
                    _ai_system.buffer.add_message("assistant", fallback_reply)
                _append_activity(_frontend_state, f"{_ai_system.config.name} 的一轮对话未完整完成")
                _save_ai()
                return _json_ok(
                    assistant={"content": fallback_reply, "bubbles": _split_bubbles(fallback_reply)},
                    snapshot=_build_snapshot(),
                    degraded=True,
                )
            except Exception:
                globals()["_ai_system"] = backup
                return _json_error(f"发送失败：{exc}", 500)

        return _json_ok(
            assistant={"content": response, "bubbles": _split_bubbles(response)},
            snapshot=_build_snapshot(),
        )


@app.post("/api/settings")
def api_settings():
    payload = request.get_json(force=True, silent=True) or {}
    new_name = (payload.get("name") or "").strip()
    persona_web_search_enabled = payload.get("personaWebSearchEnabled")
    has_name = bool(new_name)
    has_toggle = isinstance(persona_web_search_enabled, bool)
    if not has_name and not has_toggle:
        return _json_error("没有可保存的设置。")

    with _ai_lock:
        if has_name:
            _set_agent_name(new_name)
        if has_toggle:
            _frontend_state["persona_web_search_enabled"] = persona_web_search_enabled
        _save_ai()
        _save_frontend_state(_frontend_state)
        if has_name:
            _append_activity(_frontend_state, f"角色名称已更新为 {new_name}")
        if has_toggle:
            switch_text = "开启" if persona_web_search_enabled else "关闭"
            _append_activity(_frontend_state, f"人设联网补充已{switch_text}")
        return _json_ok(snapshot=_build_snapshot())


@app.post("/api/persona/text")
def api_persona_text():
    payload = request.get_json(force=True, silent=True) or {}
    text = (payload.get("text") or "").strip()
    label = (payload.get("label") or "manual_input").strip() or "manual_input"
    persona_name = (payload.get("personaName") or "").strip() or _ai_system.config.name
    work_title = (payload.get("workTitle") or "").strip()
    if not text:
        return _json_error("学习文本不能为空。")

    with _ai_lock:
        try:
            preview = _ai_system.persona_system.preview_from_sources(
                persona_name=persona_name,
                work_title=work_title,
                local_text=text,
                local_label=label,
                enable_web_search=bool(_frontend_state.get("persona_web_search_enabled", True)),
            )
        except Exception as exc:
            return _json_error(str(exc))
        _append_activity(_frontend_state, f"已根据文本资料生成 {persona_name} 的待确认预览")
        return _json_ok(preview=preview.dict(), snapshot=_build_snapshot())


@app.post("/api/persona/preview")
def api_persona_preview():
    payload = request.get_json(force=True, silent=True) or {}
    persona_name = (payload.get("personaName") or "").strip()
    work_title = (payload.get("workTitle") or "").strip()
    local_text = (payload.get("text") or "").strip()
    label = (payload.get("label") or "manual_input").strip() or "manual_input"
    if not persona_name and not local_text:
        return _json_error("角色名或资料至少提供一个。")

    with _ai_lock:
        try:
            preview = _ai_system.persona_system.preview_from_sources(
                persona_name=persona_name,
                work_title=work_title,
                local_text=local_text,
                local_label=label,
                enable_web_search=bool(_frontend_state.get("persona_web_search_enabled", True)),
            )
        except Exception as exc:
            return _json_error(str(exc))
        _append_activity(_frontend_state, f"已生成 {(persona_name or _ai_system.config.name)} 的待确认预览")
        return _json_ok(preview=preview.dict(), snapshot=_build_snapshot())


@app.post("/api/persona/confirm")
def api_persona_confirm():
    payload = request.get_json(force=True, silent=True) or {}
    preview_id = (payload.get("previewId") or "").strip()
    selected_keywords = payload.get("selectedKeywords") or []
    if not preview_id:
        return _json_error("previewId 不能为空。")

    with _ai_lock:
        try:
            result = _ai_system.persona_system.confirm_preview(preview_id, selected_keywords=selected_keywords)
        except Exception as exc:
            return _json_error(str(exc))
        preview = result["preview"]
        if preview.persona_name and preview.persona_name != _ai_system.config.name:
            _set_agent_name(preview.persona_name)
        _save_ai()
        _append_activity(_frontend_state, f"已确认写入 {preview.persona_name} 的人设预览")
        return _json_ok(count=result["count"], preview=preview.dict(), snapshot=_build_snapshot())


@app.post("/api/persona/file")
def api_persona_file():
    files = [item for item in request.files.getlist("file") if item and item.filename]
    persona_name = (request.form.get("personaName") or "").strip() or _ai_system.config.name
    work_title = (request.form.get("workTitle") or "").strip()
    if not files:
        return _json_error("没有收到资料文件。")

    saved_files = []
    local_snippets = []
    seen_local_texts = set()
    try:
        for file in files:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in ALLOWED_TEXT_SUFFIXES:
                return _json_error("仅支持 .txt / .md / .log / .json / .csv / .pdf 文件。")
            safe_name = secure_filename(file.filename) or f"persona{suffix}"
            unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
            target = UPLOAD_DIR / unique_name
            file.save(target)
            saved_files.append((target, safe_name, suffix))
        for target, safe_name, suffix in saved_files:
            text = _read_persona_text(target, suffix)
            canonical = " ".join(text.split())
            if not canonical or canonical in seen_local_texts:
                continue
            seen_local_texts.add(canonical)
            local_snippets.append({"source": "local", "title": safe_name, "text": text})
    except Exception as exc:
        return _json_error(str(exc))

    if not local_snippets:
        return _json_error("这批资料没有解析出新的有效内容。")

    with _ai_lock:
        try:
            preview = _ai_system.persona_system.preview_from_sources(
                persona_name=persona_name,
                work_title=work_title,
                local_text="",
                local_label="persona_batch",
                local_snippets=local_snippets,
                enable_web_search=bool(_frontend_state.get("persona_web_search_enabled", True)),
            )
        except Exception as exc:
            return _json_error(str(exc))
        _append_activity(_frontend_state, f"已根据 {len(local_snippets)} 份资料生成 {persona_name} 的待确认预览")
        return _json_ok(preview=preview.dict(), snapshot=_build_snapshot())


@app.post("/api/avatar")
def api_avatar():
    file = request.files.get("file")
    if file is None or not file.filename:
        return _json_error("没有收到头像文件。")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        return _json_error("仅支持 png / jpg / jpeg / webp。")

    filename = secure_filename(file.filename) or "avatar.png"
    target = AVATAR_DIR / f"avatar_{filename.rsplit('.', 1)[0]}.png"
    file.save(target)
    _ensure_avatar_square(target)
    _frontend_state["avatar_path"] = str(target)
    _save_frontend_state(_frontend_state)
    _append_activity(_frontend_state, "已更新角色头像")
    return _json_ok(snapshot=_build_snapshot())


@app.post("/api/persona/clear")
def api_clear_persona():
    with _ai_lock:
        _ai_system.persona_system.clear()
        _save_ai()
        _append_activity(_frontend_state, "已清空人设资料")
        return _json_ok(snapshot=_build_snapshot())


@app.post("/api/reset")
def api_reset():
    global _ai_system, _frontend_state
    with _ai_lock:
        for path in (Path(SAVE_PATH), Path(NEW_MEMORY_STATE_PATH), Path(NEW_PERSONA_STATE_PATH), STATE_PATH):
            if path.exists():
                path.unlink()
        _clear_uploaded_files()
        _frontend_state = _default_frontend_state()
        _save_frontend_state(_frontend_state)
        _ai_system = AISystem()
        _ai_system.on_startup()
        _save_ai()
        _append_activity(_frontend_state, "已删除全部存档并重新初始化角色")
        return _json_ok(snapshot=_build_snapshot())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
