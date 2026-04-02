import copy
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

from const import APP_DIR, PERSONA_SAVE_PATH, SAVE_PATH
from main import AIConfig, AISystem, PersonalityConfig, check_has_valid_key


BASE_DIR = Path(APP_DIR)
STATE_PATH = BASE_DIR / "frontend_state.json"
UPLOAD_DIR = BASE_DIR / "uploads"
AVATAR_DIR = UPLOAD_DIR / "avatars"
ALLOWED_TEXT_SUFFIXES = {".txt", ".md", ".log", ".json", ".csv"}
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_DIR.mkdir(parents=True, exist_ok=True)


def _default_frontend_state():
    return {
        "avatar_path": "",
        "recent_activity": [],
    }


def _load_frontend_state():
    if not STATE_PATH.exists():
        return _default_frontend_state()
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return _default_frontend_state()
    state = _default_frontend_state()
    state.update(data if isinstance(data, dict) else {})
    if not isinstance(state.get("recent_activity"), list):
        state["recent_activity"] = []
    return state


def _save_frontend_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def _append_activity(state, text):
    activity = state.setdefault("recent_activity", [])
    activity.insert(
        0,
        {
            "text": text,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    )
    del activity[10:]
    _save_frontend_state(state)


def _ensure_avatar_square(file_path):
    with Image.open(file_path) as image:
        image = image.convert("RGB")
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((512, 512))
        image.save(file_path, format="PNG")


def _split_bubbles(text):
    normalized = (text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return []
    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    return lines if len(lines) > 1 else [normalized]


def _friendliness_hearts(value):
    heart_count = max(0, min(5, int(round((value + 100) / 40))))
    return "❤" * heart_count if heart_count else "♡"


def _mood_to_zh(mood_name):
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
    }
    return mapping.get(mood_name, mood_name)


def _serialize_message(message):
    content = message.get("content")
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        content = "\n".join(part for part in text_parts if part).strip()
    if not isinstance(content, str):
        content = str(content or "")
    role = message.get("role", "assistant")
    bubbles = _split_bubbles(content) if role == "assistant" else [content.strip()] if content.strip() else []
    return {
        "role": role,
        "content": content,
        "bubbles": bubbles,
    }


def _has_user_conversation():
    return any(message.get("role") == "user" for message in _ai_system.get_message_history(False))


def _bootstrap_ai():
    ai = AISystem.load(SAVE_PATH)
    is_new = ai is None
    if is_new:
        ai = AISystem()
        ai.on_startup()
    else:
        ai.on_startup()
    return ai


app = Flask(__name__, template_folder="templates", static_folder="static")
_ai_lock = Lock()
_frontend_state = _load_frontend_state()
_ai_system = _bootstrap_ai() if check_has_valid_key() else AISystem()


def _save_ai():
    _ai_system.save(SAVE_PATH)


def _set_agent_name(new_name):
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


def _build_snapshot():
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
    has_user_conversation = _has_user_conversation()
    mood_name = _mood_to_zh(_ai_system.emotion_system.get_mood_name()) if has_user_conversation else "平静"
    affinity = _friendliness_hearts(_ai_system.relation_system.friendliness) if has_user_conversation else "♡"
    friendliness = round(_ai_system.relation_system.friendliness, 2) if has_user_conversation else 0.0
    mood_description = _ai_system.emotion_system.get_mood_prompt() if has_user_conversation else "情绪稳定，尚未因对话产生明显波动。"
    return {
        "agent": {
            "name": _ai_system.config.name,
            "avatarUrl": avatar_url,
            "mood": mood_name,
            "moodDescription": mood_description,
            "friendliness": friendliness,
            "affinity": affinity,
            "personaChunks": _ai_system.persona_system.chunk_count,
            "keywords": _ai_system.persona_system.get_display_keywords(),
            "personaStatus": _ai_system.persona_system.get_status().dict(),
        },
        "history": history,
        "recentActivity": _frontend_state.get("recent_activity", []),
    }


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
        return jsonify({"ok": False, "error": "消息不能为空。"}), 400
    with _ai_lock:
        backup = copy.deepcopy(_ai_system)
        try:
            response = _ai_system.send_message(message)
            _save_ai()
            _append_activity(_frontend_state, f"与 {_ai_system.config.name} 完成了一轮对话")
        except Exception as exc:
            traceback.print_exc()
            fallback_reply = "刚刚这条消息我没能顺利接住，可能是网络或模型服务暂时不稳定。你可以再发一次，我会继续。"
            try:
                if _ai_system.get_message_history(False) and _ai_system.get_message_history(False)[-1].get("role") != "assistant":
                    _ai_system.buffer.add_message("assistant", fallback_reply)
                _append_activity(_frontend_state, f"与 {_ai_system.config.name} 的一轮对话未完整完成")
                _save_ai()
                return jsonify(
                    {
                        "ok": True,
                        "assistant": {
                            "content": fallback_reply,
                            "bubbles": _split_bubbles(fallback_reply),
                        },
                        "snapshot": _build_snapshot(),
                        "degraded": True,
                    }
                )
            except Exception:
                globals()["_ai_system"] = backup
                return jsonify({"ok": False, "error": f"发送失败：{exc}"}), 500
        return jsonify(
            {
                "ok": True,
                "assistant": {
                    "content": response,
                    "bubbles": _split_bubbles(response),
                },
                "snapshot": _build_snapshot(),
            }
        )


@app.post("/api/settings")
def api_settings():
    payload = request.get_json(force=True, silent=True) or {}
    new_name = (payload.get("name") or "").strip()
    if not new_name:
        return jsonify({"ok": False, "error": "名字不能为空。"}), 400
    with _ai_lock:
        _set_agent_name(new_name)
        _save_ai()
        _append_activity(_frontend_state, f"角色名称已更新为 {new_name}")
        return jsonify({"ok": True, "snapshot": _build_snapshot()})


@app.post("/api/persona/text")
def api_persona_text():
    payload = request.get_json(force=True, silent=True) or {}
    text = (payload.get("text") or "").strip()
    label = (payload.get("label") or "manual_input").strip() or "manual_input"
    persona_name = (payload.get("personaName") or "").strip() or _ai_system.config.name
    work_title = (payload.get("workTitle") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "学习文本不能为空。"}), 400
    with _ai_lock:
        try:
            preview = _ai_system.persona_system.preview_from_sources(
                persona_name=persona_name,
                work_title=work_title,
                local_text=text,
                local_label=label,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        _append_activity(_frontend_state, f"已根据文本资料生成 {persona_name} 的待确认预览")
        return jsonify({"ok": True, "preview": preview.dict(), "snapshot": _build_snapshot()})


@app.post("/api/persona/preview")
def api_persona_preview():
    payload = request.get_json(force=True, silent=True) or {}
    persona_name = (payload.get("personaName") or "").strip()
    work_title = (payload.get("workTitle") or "").strip()
    local_text = (payload.get("text") or "").strip()
    label = (payload.get("label") or "manual_input").strip() or "manual_input"
    if not persona_name and not local_text:
        return jsonify({"ok": False, "error": "角色名或资料至少提供一个。"}), 400
    with _ai_lock:
        try:
            preview = _ai_system.persona_system.preview_from_sources(
                persona_name=persona_name,
                work_title=work_title,
                local_text=local_text,
                local_label=label,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        _append_activity(_frontend_state, f"已生成 {(persona_name or _ai_system.config.name)} 的待确认预览")
        return jsonify({"ok": True, "preview": preview.dict(), "snapshot": _build_snapshot()})


@app.post("/api/persona/confirm")
def api_persona_confirm():
    payload = request.get_json(force=True, silent=True) or {}
    preview_id = (payload.get("previewId") or "").strip()
    if not preview_id:
        return jsonify({"ok": False, "error": "previewId 不能为空。"}), 400
    with _ai_lock:
        try:
            result = _ai_system.persona_system.confirm_preview(preview_id)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        preview = result["preview"]
        if preview.persona_name and preview.persona_name != _ai_system.config.name:
            _set_agent_name(preview.persona_name)
        _save_ai()
        _append_activity(_frontend_state, f"已确认写入 {preview.persona_name} 的冷启动人设")
        return jsonify(
            {
                "ok": True,
                "count": result["count"],
                "preview": preview.dict(),
                "snapshot": _build_snapshot(),
            }
        )


@app.post("/api/persona/file")
def api_persona_file():
    file = request.files.get("file")
    persona_name = (request.form.get("personaName") or "").strip() or _ai_system.config.name
    work_title = (request.form.get("workTitle") or "").strip()
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "没有收到文件。"}), 400
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_TEXT_SUFFIXES:
        return jsonify({"ok": False, "error": "仅支持 .txt .md .log .json .csv 文件。"}), 400
    safe_name = secure_filename(file.filename) or f"persona{suffix}"
    target = UPLOAD_DIR / safe_name
    file.save(target)
    with open(target, "r", encoding="utf-8", errors="ignore") as saved_file:
        text = saved_file.read()
    with _ai_lock:
        try:
            preview = _ai_system.persona_system.preview_from_sources(
                persona_name=persona_name,
                work_title=work_title,
                local_text=text,
                local_label=safe_name,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        _append_activity(_frontend_state, f"已根据文件资料生成 {persona_name} 的待确认预览")
        return jsonify({"ok": True, "preview": preview.dict(), "snapshot": _build_snapshot()})


@app.post("/api/avatar")
def api_avatar():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "没有收到头像文件。"}), 400
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        return jsonify({"ok": False, "error": "仅支持 png / jpg / jpeg / webp。"}), 400
    filename = secure_filename(file.filename) or "avatar.png"
    target = AVATAR_DIR / f"avatar_{filename.rsplit('.', 1)[0]}.png"
    file.save(target)
    _ensure_avatar_square(target)
    _frontend_state["avatar_path"] = str(target)
    _save_frontend_state(_frontend_state)
    _append_activity(_frontend_state, "已更新角色头像")
    return jsonify({"ok": True, "snapshot": _build_snapshot()})


@app.post("/api/persona/clear")
def api_clear_persona():
    with _ai_lock:
        _ai_system.persona_system.clear()
        _save_ai()
        _append_activity(_frontend_state, "已清空人设资料")
        return jsonify({"ok": True, "snapshot": _build_snapshot()})


@app.post("/api/reset")
def api_reset():
    global _ai_system, _frontend_state
    with _ai_lock:
        for path in (Path(SAVE_PATH), Path(PERSONA_SAVE_PATH), STATE_PATH):
            if path.exists():
                path.unlink()
        if AVATAR_DIR.exists():
            for file in AVATAR_DIR.iterdir():
                if file.is_file():
                    file.unlink()
        _frontend_state = _default_frontend_state()
        _save_frontend_state(_frontend_state)
        _ai_system = AISystem()
        _ai_system.on_startup()
        _save_ai()
        _append_activity(_frontend_state, "已删除全部存档并重新初始化角色")
        return jsonify(
            {
                "ok": True,
                "snapshot": _build_snapshot(),
            }
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
