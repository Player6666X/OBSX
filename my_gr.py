import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import time
import subprocess
import os
import sys
import json
import tempfile
import shutil
import signal
import atexit
import logging
import argparse
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image, ImageTk

# ===== Opcionais =====
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    ctk = None
    CTK_AVAILABLE = False
    print("💡 Para uma interface mais moderna, instale: pip install customtkinter")

try:
    import dxcam
    DXCAM_AVAILABLE = True
except Exception:
    dxcam = None
    DXCAM_AVAILABLE = False

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except Exception:
    keyboard = None
    KEYBOARD_AVAILABLE = False

try:
    import pyaudiowpatch as pyaudio
    AUDIO_AVAILABLE = True
except Exception:
    try:
        import pyaudio
        AUDIO_AVAILABLE = True
    except Exception:
        pyaudio = None
        AUDIO_AVAILABLE = False


SETTINGS_FILE = "obsx_config.json"
LOG_FILE = "obsx.log"
PLUGINS_DIR = "plugins"

DEFAULT_SETTINGS = {
    "output_dir": os.getcwd(),
    "fps": 60,
    "reencode": False,
    "mic_enabled": False,
    "monitor_index": 0,
    "region": None,
    "resize_resolution": None,
    "video_codec": "auto",
    "bitrate": "192k",
    "video_bitrate": "",
    "start_hotkey": "f12",
    "pause_hotkey": "f11",
    "mute_hotkey": "",
    "schedule_start": "",
    "schedule_stop": "",
    "show_indicator": True,
    "game_mode": True,
    "low_latency": False,
    "stream_enabled": False,
    "stream_url": "",
    "stream_key": "",
    "stream_bitrate": "2000k",
    "stream_resolution": "1280x720",
    "overlay_text": "",
    "overlay_image": "",
    "overlay_pos": "top-left",
    "audio_device_loopback": None,
    "audio_device_mic": None,
    "plugins_enabled": False,
    "preview_size": [320, 180],
    "schedule_active": False,
    "remux_mp4": True,
    "encoder_preset": "veryfast",
    "audio_sample_rate": 48000,
}


def _to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "sim")
    return bool(value)


def _normalize_settings(data):
    clean = {}
    for k, v in data.items():
        key = str(k).strip().replace(" ", "_")
        clean[key] = v

    merged = {**DEFAULT_SETTINGS, **clean}

    # Diretório
    merged["output_dir"] = str(merged.get("output_dir") or os.getcwd()).strip() or os.getcwd()

    # Inteiros
    for key in ("fps", "audio_sample_rate"):
        try:
            merged[key] = int(merged.get(key, DEFAULT_SETTINGS[key]))
        except Exception:
            merged[key] = DEFAULT_SETTINGS[key]

    # Booleanos
    bool_keys = [
        "reencode",
        "mic_enabled",
        "show_indicator",
        "game_mode",
        "low_latency",
        "stream_enabled",
        "plugins_enabled",
        "schedule_active",
        "remux_mp4",
    ]
    for key in bool_keys:
        merged[key] = _to_bool(merged.get(key, DEFAULT_SETTINGS[key]))

    # Strings
    str_keys = [
        "video_codec",
        "bitrate",
        "video_bitrate",
        "start_hotkey",
        "pause_hotkey",
        "mute_hotkey",
        "schedule_start",
        "schedule_stop",
        "stream_url",
        "stream_key",
        "stream_bitrate",
        "stream_resolution",
        "overlay_text",
        "overlay_image",
        "overlay_pos",
        "encoder_preset",
    ]
    for key in str_keys:
        value = merged.get(key, DEFAULT_SETTINGS[key])
        merged[key] = "" if value is None else str(value).strip()

    # Preview size
    preview = merged.get("preview_size", DEFAULT_SETTINGS["preview_size"])
    if isinstance(preview, (list, tuple)) and len(preview) == 2:
        try:
            merged["preview_size"] = [int(preview[0]), int(preview[1])]
        except Exception:
            merged["preview_size"] = [320, 180]
    else:
        merged["preview_size"] = [320, 180]

    # Resize resolution
    resize = merged.get("resize_resolution")
    if isinstance(resize, (list, tuple)) and len(resize) == 2:
        try:
            merged["resize_resolution"] = [int(resize[0]), int(resize[1])]
        except Exception:
            merged["resize_resolution"] = None
    else:
        merged["resize_resolution"] = None

    # Região
    region = merged.get("region")
    if not (
        isinstance(region, dict)
        and all(k in region for k in ("x", "y", "w", "h", "monitor"))
    ):
        merged["region"] = None

    return merged


def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _normalize_settings(data)
    except Exception:
        return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    tmp_path = SETTINGS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, SETTINGS_FILE)


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("obsx")


class FFmpegWriter:
    """
    Writer de vídeo usando FFmpeg direto, com entrada rawvideo via stdin.
    Mais adequado para um gravador estilo OBS do que cv2.VideoWriter.
    """

    def __init__(self, path, fps, size, encoder_args):
        self.path = path
        self.fps = fps
        self.size = size
        self.process = None

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{size[0]}x{size[1]}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "pipe:0",
            *encoder_args,
            "-f", "matroska",
            path,
        ]

        popen_kwargs = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self.process = subprocess.Popen(cmd, **popen_kwargs)

        if self.process.poll() is not None:
            raise RuntimeError("FFmpeg não conseguiu iniciar o writer.")

    def isOpened(self):
        return self.process is not None and self.process.poll() is None

    def write(self, frame):
        if not self.isOpened():
            return
        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            self.release()

    def release(self):
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
            except Exception:
                pass

            try:
                self.process.wait(timeout=15)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

            self.process = None


class ScreenRecorder:
    def __init__(self, **kwargs):
        self.settings = load_settings()
        for k, v in kwargs.items():
            if v is not None:
                self.settings[k] = v
        self.settings = _normalize_settings(self.settings)

        # Configurações principais
        self.fps = int(self.settings.get("fps", 60))
        self.reencode = bool(self.settings.get("reencode", False))
        self.output_dir = str(self.settings.get("output_dir", os.getcwd()))
        self.mic_enabled = bool(self.settings.get("mic_enabled", False))
        self.monitor_index = int(self.settings.get("monitor_index", 0))
        self.region = self.settings.get("region")
        resize = self.settings.get("resize_resolution")
        self.resize_resolution = tuple(resize) if isinstance(resize, (list, tuple)) and len(resize) == 2 else None
        self.video_codec = str(self.settings.get("video_codec", "auto"))
        self.bitrate = str(self.settings.get("bitrate", "192k"))
        self.video_bitrate = str(self.settings.get("video_bitrate", ""))
        self.start_hotkey = str(self.settings.get("start_hotkey", "f12"))
        self.pause_hotkey = str(self.settings.get("pause_hotkey", "f11"))
        self.mute_hotkey = str(self.settings.get("mute_hotkey", ""))
        self.schedule_start = str(self.settings.get("schedule_start", ""))
        self.schedule_stop = str(self.settings.get("schedule_stop", ""))
        self.show_indicator = bool(self.settings.get("show_indicator", True))
        self.game_mode = bool(self.settings.get("game_mode", True))
        self.low_latency = bool(self.settings.get("low_latency", False))
        self.stream_enabled = bool(self.settings.get("stream_enabled", False))
        self.stream_url = str(self.settings.get("stream_url", ""))
        self.stream_key = str(self.settings.get("stream_key", ""))
        self.stream_bitrate = str(self.settings.get("stream_bitrate", "2000k"))
        self.stream_resolution = str(self.settings.get("stream_resolution", "1280x720"))
        self.overlay_text = str(self.settings.get("overlay_text", ""))
        self.overlay_image = str(self.settings.get("overlay_image", ""))
        self.overlay_pos = str(self.settings.get("overlay_pos", "top-left"))
        self.audio_device_loopback = self.settings.get("audio_device_loopback")
        self.audio_device_mic = self.settings.get("audio_device_mic")
        self.plugins_enabled = bool(self.settings.get("plugins_enabled", False))
        self.preview_size = tuple(self.settings.get("preview_size", [320, 180]))
        self.schedule_active = bool(self.settings.get("schedule_active", False))
        self.remux_mp4 = bool(self.settings.get("remux_mp4", True))
        self.encoder_preset = str(self.settings.get("encoder_preset", "veryfast"))
        self.audio_target_rate = int(self.settings.get("audio_sample_rate", 48000))

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(PLUGINS_DIR, exist_ok=True)

        # Estado
        self.recording = False
        self.paused = False
        self.pause_lock = threading.Lock()
        self.camera = None
        self.video_writer = None
        self.video_path = None
        self.output_path = None
        self.current_output_size = None

        self.stream_process = None

        self.audio_stream = None
        self.audio_file = None
        self.audio_file_path = None
        self.mic_stream = None
        self.mic_file = None
        self.mic_file_path = None
        self.mic_muted = False
        self.p_audio = None
        self.audio_channels = 2
        self.audio_rate = 48000

        self.video_thread = None
        self.stop_event = threading.Event()
        self.start_time = None
        self._elapsed_before_pause = 0.0

        self.loopback_rms = 0.0
        self.mic_rms = 0.0
        self.smoothed_loopback = 0.0
        self.smoothed_mic = 0.0
        self.rms_lock = threading.Lock()

        self._lock = threading.Lock()
        self._toggling = False
        self._cleaned = False

        self.hotkeys_registered = []
        self.schedule_thread = None
        self.schedule_stop_event = threading.Event()

        self.plugin_objects = []
        self._overlay_cache = {}

        self.last_preview_frame = None

        self.ffmpeg_available = self._check_ffmpeg()

        self._init_ui()
        self._register_hotkeys()

        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self._cleanup)

        if self.plugins_enabled:
            self._load_plugins()

        self._update_monitor_list()
        self._update_audio_devices()

        if not self.ffmpeg_available:
            self._set_combobox_values(self.codec_combo, ["copy"])
            self.codec_combo.set("copy")
            self.video_codec = "copy"
            if hasattr(self, "ffmpeg_warning_label"):
                self.ffmpeg_warning_label.configure(text="⚠️ FFmpeg não encontrado!")

        if self.schedule_active and self.schedule_start and self.schedule_stop:
            self._restore_schedule()

        self.root.after(100, self._update_vu_meters)

    # ========== Helpers de thread/UI ==========
    def _ui(self, callback, *args, **kwargs):
        try:
            if not hasattr(self, "root") or self.root is None:
                return
            if threading.current_thread() is threading.main_thread():
                callback(*args, **kwargs)
            else:
                self.root.after(0, lambda: callback(*args, **kwargs))
        except Exception:
            logger.exception("Erro ao executar callback de UI")

    def _show_error(self, title, message):
        self._ui(lambda: messagebox.showerror(title, message))

    def _show_info(self, title, message):
        self._ui(lambda: messagebox.showinfo(title, message))

    def _log(self, msg, level="info"):
        try:
            log_func = getattr(logger, level, logger.info)
            log_func(msg)
        except Exception:
            pass

        def _append():
            if not hasattr(self, "log_text"):
                return
            try:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

                # Limita o tamanho do log
                try:
                    line_count = int(self.log_text.index("end-1c").split(".")[0])
                    if line_count > 1000:
                        self.log_text.delete("1.0", "200.0")
                except Exception:
                    pass

                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            except Exception:
                pass

        self._ui(_append)

    def _set_entry(self, entry, text):
        def _update():
            try:
                entry.configure(state="normal")
            except Exception:
                pass
            entry.delete(0, "end")
            entry.insert(0, str(text))
            try:
                entry.configure(state="disabled")
            except Exception:
                pass

        self._ui(_update)

    def _set_combobox_values(self, combo, values):
        try:
            if CTK_AVAILABLE:
                combo.configure(values=values)
            else:
                combo["values"] = values
        except Exception:
            pass

    # ========== UI ==========
    def _init_ui(self):
        if CTK_AVAILABLE:
            ctk.set_appearance_mode("System")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
            self.root.title("OBSX - Gravador Profissional")
            self.root.geometry("1150x820")
            base_frame = ctk.CTkFrame(self.root)
            base_frame.pack(fill="both", expand=True, padx=10, pady=10)
        else:
            self.root = tk.Tk()
            self.root.title("OBSX - Gravador Profissional")
            self.root.geometry("1150x820")
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass
            base_frame = ttk.Frame(self.root)
            base_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Painel esquerdo
        if CTK_AVAILABLE:
            settings_frame = ctk.CTkFrame(base_frame)
        else:
            settings_frame = ttk.LabelFrame(base_frame, text="Configurações")
        settings_frame.pack(side="left", fill="y", padx=5, pady=5, ipadx=5, ipady=5)

        if not self.ffmpeg_available:
            if CTK_AVAILABLE:
                self.ffmpeg_warning_label = ctk.CTkLabel(
                    settings_frame,
                    text="⚠️ FFmpeg não encontrado!",
                    text_color="red",
                )
            else:
                self.ffmpeg_warning_label = tk.Label(
                    settings_frame,
                    text="⚠️ FFmpeg não encontrado!",
                    fg="red",
                )
            self.ffmpeg_warning_label.pack(anchor="w", pady=2)

        # Abas
        if CTK_AVAILABLE:
            notebook = ctk.CTkTabview(settings_frame)
            notebook.pack(fill="both", expand=True, pady=5)
            notebook.add("Geral")
            notebook.add("Áudio")
            notebook.add("Stream")
            notebook.add("Overlay")
            notebook.add("Plugins")

            self.tab_frames = {
                "Geral": notebook.tab("Geral"),
                "Áudio": notebook.tab("Áudio"),
                "Stream": notebook.tab("Stream"),
                "Overlay": notebook.tab("Overlay"),
                "Plugins": notebook.tab("Plugins"),
            }
        else:
            notebook = ttk.Notebook(settings_frame)
            notebook.pack(fill="both", expand=True, pady=5)
            self.tab_frames = {}
            for name in ["Geral", "Áudio", "Stream", "Overlay", "Plugins"]:
                f = ttk.Frame(notebook)
                notebook.add(f, text=name)
                self.tab_frames[name] = f

        self._populate_geral(self.tab_frames["Geral"])
        self._populate_audio(self.tab_frames["Áudio"])
        self._populate_stream(self.tab_frames["Stream"])
        self._populate_overlay(self.tab_frames["Overlay"])
        self._populate_plugins(self.tab_frames["Plugins"])

        self.save_btn = self._add_button(
            settings_frame,
            "Salvar Config.",
            self._save_current_settings,
            width=15,
            pack=False,
        )
        self.save_btn.pack(pady=10, fill="x")

        # Painel direito
        if CTK_AVAILABLE:
            right_frame = ctk.CTkFrame(base_frame)
        else:
            right_frame = ttk.Frame(base_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        btn_frame = self._add_frame(right_frame)

        self.btn = self._add_button(
            btn_frame,
            "▶ GRAVAR",
            self.toggle_recording,
            width=15,
            big=True,
            pack=False,
        )
        self.btn.pack(side="left", padx=5)

        self.pause_btn = self._add_button(
            btn_frame,
            "⏸ PAUSAR",
            self.toggle_pause,
            width=15,
            state="disabled",
            pack=False,
        )
        self.pause_btn.pack(side="left", padx=5)

        if self.mic_enabled:
            self.mic_btn = self._add_button(
                btn_frame,
                "🎤 Ligado",
                self.toggle_mute_mic,
                width=15,
                state="disabled",
                pack=False,
            )
            self.mic_btn.pack(side="left", padx=5)
        else:
            self.mic_btn = None

        self.label_time = self._add_label(right_frame, "00:00", font_size=18)

        vu_frame = self._add_frame(right_frame)
        self._add_label(vu_frame, "Loopback:", small=True)
        self.loopback_vu = self._add_progressbar(vu_frame, length=180)
        self._add_label(vu_frame, "Mic:", small=True)
        self.mic_vu = self._add_progressbar(vu_frame, length=180)

        preview_frame = self._add_frame(right_frame)
        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=self.preview_size[0],
            height=self.preview_size[1],
            bg="black",
            highlightthickness=0,
        )
        self.preview_canvas.pack()
        self.preview_img = None
        self.preview_canvas.bind("<Configure>", self._on_preview_resize)

        self._add_label(
            right_frame,
            "⚠️ Minimize a janela para não aparecer na gravação",
            font_size=9,
            fg="red",
        )

        log_frame = self._add_frame(right_frame)
        self._add_label(log_frame, "Log:", small=True)
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=6,
            width=64,
            state="disabled",
            font=("Consolas", 8),
        )
        self.log_text.pack(fill="both", expand=True)

        self._log("Aplicação iniciada.")

    def _populate_geral(self, parent):
        self._add_label(parent, "Diretório de saída:")
        dir_frame = self._add_frame(parent)
        self.dir_entry = self._add_entry(dir_frame, self.output_dir, width=25, pack=False)
        self.dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._add_button(dir_frame, "Procurar", self._browse_output_dir, width=8, side="left")
        self._add_button(dir_frame, "Abrir", self._open_output_dir, width=6, side="left")

        self._add_label(parent, "FPS:")
        self.fps_combo = self._add_combobox(parent, ["15", "30", "60", "120"], str(self.fps), self._on_fps_change)

        self._add_label(parent, "Resolução:")
        res_values = ["Nativa", "1920x1080", "1280x720", "854x480"]
        self.res_combo = self._add_combobox(
            parent,
            res_values,
            self._res_to_str(self.resize_resolution),
            self._on_res_change,
        )

        self._add_label(parent, "Codec:")
        codec_values = ["auto", "h264", "h265", "copy"] if self.ffmpeg_available else ["copy"]
        initial_codec = self.video_codec if self.video_codec in codec_values else "copy"
        self.codec_combo = self._add_combobox(parent, codec_values, initial_codec, self._on_codec_change)

        self._add_label(parent, "Preset do encoder:")
        self.encoder_preset_combo = self._add_combobox(
            parent,
            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
            self.encoder_preset,
            self._on_encoder_preset_change,
        )

        self._add_label(parent, "Bitrate de áudio:")
        self.audio_bitrate_entry = self._add_entry(parent, self.bitrate, width=10)

        self._add_label(parent, "Bitrate de vídeo:")
        self.video_bitrate_entry = self._add_entry(parent, self.video_bitrate or "auto", width=10)

        self._add_label(parent, "Taxa de áudio (Hz):")
        self.audio_rate_combo = self._add_combobox(
            parent,
            ["44100", "48000"],
            str(self.audio_target_rate),
            self._on_audio_target_rate_change,
        )

        self.remux_var = tk.BooleanVar(value=self.remux_mp4)
        self.remux_check = self._add_checkbox(
            parent,
            "Converter para MP4 após gravar (remux)",
            self.remux_var,
            self._on_remux_toggle,
        )

        self.mic_var = tk.BooleanVar(value=self.mic_enabled)
        self.mic_check = self._add_checkbox(parent, "Microfone", self.mic_var, self._on_mic_toggle)

        self.indicator_var = tk.BooleanVar(value=self.show_indicator)
        self.indicator_check = self._add_checkbox(parent, "Indicador de gravação", self.indicator_var, None)

        self.game_mode_var = tk.BooleanVar(value=self.game_mode)
        self.game_mode_check = self._add_checkbox(
            parent,
            "Modo Jogo (video_mode)",
            self.game_mode_var,
            lambda: setattr(self, "game_mode", self.game_mode_var.get()),
        )

        self.low_latency_var = tk.BooleanVar(value=self.low_latency)
        self.low_latency_check = self._add_checkbox(
            parent,
            "Baixa Latência",
            self.low_latency_var,
            lambda: setattr(self, "low_latency", self.low_latency_var.get()),
        )

        self._add_label(parent, "Monitor:")
        self.monitor_combo = self._add_combobox(parent, ["Carregando..."], "", self._on_monitor_change)

        self._add_label(parent, "Região:")
        region_frame = self._add_frame(parent)
        self.region_entry = self._add_entry(
            region_frame,
            self._region_to_str(self.region),
            width=18,
            state="readonly",
            pack=False,
        )
        self.region_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._add_button(region_frame, "Selecionar", self._select_region, width=9, side="left")

        self._add_label(parent, "Atalhos:")
        hk_frame = self._add_frame(parent)

        self._add_label(hk_frame, "Iniciar/Parar:", small=True)
        self.start_hk_entry = self._add_entry(hk_frame, self.start_hotkey, width=7)

        self._add_label(hk_frame, "Pausar:", small=True)
        self.pause_hk_entry = self._add_entry(hk_frame, self.pause_hotkey, width=7)

        self._add_label(hk_frame, "Mudo:", small=True)
        self.mute_hk_entry = self._add_entry(hk_frame, self.mute_hotkey or "", width=7)

        self._add_button(hk_frame, "Aplicar", self._apply_hotkeys, width=8)

        self._add_label(parent, "Agendamento:")
        sched_frame = self._add_frame(parent)

        self._add_label(sched_frame, "Início (HH:MM):", small=True)
        self.sched_start_entry = self._add_entry(sched_frame, self.schedule_start, width=8)

        self._add_label(sched_frame, "Fim (HH:MM):", small=True)
        self.sched_stop_entry = self._add_entry(sched_frame, self.schedule_stop, width=8)

        self.sched_btn = self._add_button(sched_frame, "Ativar", self._toggle_schedule, width=8)

    def _populate_audio(self, parent):
        self._add_label(parent, "Dispositivo Loopback (áudio do sistema):")
        self.loopback_dev_combo = self._add_combobox(parent, ["Padrão"], "Padrão", self._on_loopback_dev_change)

        self._add_label(parent, "Dispositivo Microfone:")
        self.mic_dev_combo = self._add_combobox(parent, ["Padrão"], "Padrão", self._on_mic_dev_change)

        self._add_button(parent, "Atualizar dispositivos", self._update_audio_devices, width=22)

    def _populate_stream(self, parent):
        self.stream_enabled_var = tk.BooleanVar(value=self.stream_enabled)
        self.stream_check = self._add_checkbox(
            parent,
            "Habilitar Streaming",
            self.stream_enabled_var,
            lambda: setattr(self, "stream_enabled", self.stream_enabled_var.get()),
        )

        self._add_label(parent, "URL do servidor RTMP:")
        self.stream_url_entry = self._add_entry(parent, self.stream_url, width=42)

        self._add_label(parent, "Chave de stream:")
        self.stream_key_entry = self._add_entry(parent, self.stream_key, width=42)

        self._add_label(parent, "Bitrate de stream (vídeo):")
        self.stream_bitrate_entry = self._add_entry(parent, self.stream_bitrate, width=12)

        self._add_label(parent, "Resolução de stream:")
        self.stream_res_entry = self._add_entry(parent, self.stream_resolution, width=12)

        self._add_button(parent, "Testar Stream", self._test_stream, width=16)

    def _populate_overlay(self, parent):
        self._add_label(parent, "Texto sobreposto:")
        self.overlay_text_entry = self._add_entry(parent, self.overlay_text, width=42)
        self._add_button(
            parent,
            "Atualizar Texto",
            lambda: setattr(self, "overlay_text", self.overlay_text_entry.get()),
            width=16,
        )

        self._add_label(parent, "Imagem de overlay (caminho):")
        self.overlay_image_entry = self._add_entry(parent, self.overlay_image, width=42)

        img_btn_frame = self._add_frame(parent)
        self._add_button(img_btn_frame, "Carregar Imagem", self._browse_overlay_image, width=16, side="left")
        self._add_button(
            img_btn_frame,
            "Atualizar Caminho",
            lambda: setattr(self, "overlay_image", self.overlay_image_entry.get()),
            width=16,
            side="left",
        )

        self._add_label(parent, "Posição:")
        self.overlay_pos_combo = self._add_combobox(
            parent,
            ["top-left", "top-right", "bottom-left", "bottom-right"],
            self.overlay_pos,
            lambda v: setattr(self, "overlay_pos", v),
        )

    def _populate_plugins(self, parent):
        self.plugins_enabled_var = tk.BooleanVar(value=self.plugins_enabled)
        self.plugins_check = self._add_checkbox(
            parent,
            "Habilitar Plugins",
            self.plugins_enabled_var,
            self._on_plugins_toggle,
        )

        self._add_label(parent, "Plugins carregados:")
        self.plugins_listbox = tk.Listbox(parent, height=6)
        self.plugins_listbox.pack(fill="x", pady=5)

        self._refresh_plugins_list()
        self._add_button(parent, "Recarregar Plugins", self._load_plugins, width=22)

    # ---- Widgets helpers ----
    def _add_label(self, parent, text, font_size=10, fg=None, bg=None, small=False, **kwargs):
        if small:
            font_size = max(8, font_size - 1)

        if CTK_AVAILABLE:
            label_kwargs = dict(font=("Arial", font_size))
            if fg:
                label_kwargs["text_color"] = fg
            label_kwargs.update(kwargs)
            lbl = ctk.CTkLabel(parent, text=text, **label_kwargs)
        else:
            lbl = tk.Label(parent, text=text, font=("Arial", font_size), fg=fg, bg=bg, **kwargs)

        lbl.pack(anchor="w", pady=2)
        return lbl

    def _add_entry(self, parent, text, width=20, state="normal", pack=True, side="top", fill="x", padx=0, pady=2, expand=False, **kwargs):
        final_state = "disabled" if state in ("readonly", "disabled") else state

        if CTK_AVAILABLE:
            entry = ctk.CTkEntry(parent, width=width * 10, **kwargs)
            entry.insert(0, str(text))
            if final_state == "disabled":
                entry.configure(state="disabled")
        else:
            entry = tk.Entry(parent, width=width, state="normal", **kwargs)
            entry.insert(0, str(text))
            entry.configure(state=final_state)

        if pack:
            entry.pack(side=side, fill=fill, padx=padx, pady=pady, expand=expand)
        return entry

    def _add_button(
        self,
        parent,
        text,
        command,
        width=10,
        big=False,
        state="normal",
        pack=True,
        side="left",
        padx=2,
        pady=2,
        fill=None,
        anchor=None,
        **kwargs,
    ):
        if CTK_AVAILABLE:
            btn = ctk.CTkButton(
                parent,
                text=text,
                command=command,
                width=width * 10,
                height=40 if big else 28,
                state=state,
                **kwargs,
            )
        else:
            btn = tk.Button(
                parent,
                text=text,
                command=command,
                width=width,
                height=2 if big else 1,
                state=state,
                **kwargs,
            )

        if pack:
            btn.pack(side=side, padx=padx, pady=pady, fill=fill, anchor=anchor)
        return btn

    def _add_combobox(self, parent, values, initial, callback):
        if CTK_AVAILABLE:
            combo = ctk.CTkComboBox(parent, values=values, command=callback)
            combo.set(initial)
        else:
            combo = ttk.Combobox(parent, values=values, state="readonly")
            combo.set(initial)
            if callback:
                combo.bind("<<ComboboxSelected>>", lambda e: callback(combo.get()))

        combo.pack(anchor="w", pady=2, fill="x")
        return combo

    def _add_checkbox(self, parent, text, var, command):
        if CTK_AVAILABLE:
            cb = ctk.CTkCheckBox(parent, text=text, variable=var, command=command)
        else:
            cb = tk.Checkbutton(parent, text=text, variable=var, command=command)
        cb.pack(anchor="w", pady=2)
        return cb

    def _add_frame(self, parent):
        if CTK_AVAILABLE:
            f = ctk.CTkFrame(parent)
        else:
            f = tk.Frame(parent)
        f.pack(anchor="w", pady=5, fill="x")
        return f

    def _add_progressbar(self, parent, length=150):
        if CTK_AVAILABLE:
            bar = ctk.CTkProgressBar(parent, width=length, height=15, mode="determinate")
            bar.set(0)
        else:
            bar = ttk.Progressbar(parent, orient="horizontal", length=length, mode="determinate")
            bar["maximum"] = 100
        bar.pack(anchor="w", pady=2)
        return bar

    # ========== Callbacks de configuração ==========
    def _browse_output_dir(self):
        d = filedialog.askdirectory(initialdir=self.output_dir)
        if d:
            self.output_dir = d
            self._update_dir_entry()
            self._log(f"Diretório alterado para {d}")

    def _open_output_dir(self):
        path = self.dir_entry.get().strip() or self.output_dir
        if not os.path.isdir(path):
            messagebox.showerror("Erro", "Diretório inválido.")
            return

        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            self._log("Falha ao abrir diretório", "error")

    def _update_dir_entry(self):
        self.dir_entry.delete(0, tk.END)
        self.dir_entry.insert(0, self.output_dir)

    def _on_fps_change(self, val):
        try:
            self.fps = int(str(val).strip())
            self._log(f"FPS alterado para {self.fps}")
        except Exception:
            pass

    def _res_to_str(self, res):
        if not res:
            return "Nativa"
        w, h = res
        if w == 1920 and h == 1080:
            return "1920x1080"
        if w == 1280 and h == 720:
            return "1280x720"
        if w == 854 and h == 480:
            return "854x480"
        return f"{w}x{h}"

    def _on_res_change(self, val):
        if val == "Nativa":
            self.resize_resolution = None
        else:
            try:
                w, h = map(int, str(val).lower().split("x"))
                w -= w % 2
                h -= h % 2
                self.resize_resolution = (w, h)
            except Exception:
                self.resize_resolution = None
        self._log(f"Resolução alterada para {val}")

    def _on_codec_change(self, val):
        if val in ("auto", "h264", "h265") and not self.ffmpeg_available:
            self._show_error("FFmpeg ausente", "FFmpeg não encontrado. Usando 'copy'.")
            self.codec_combo.set("copy")
            val = "copy"

        self.video_codec = str(val)
        self._log(f"Codec alterado para {val}")

    def _on_encoder_preset_change(self, val):
        self.encoder_preset = str(val)
        self._log(f"Preset de encoder alterado para {val}")

    def _on_audio_target_rate_change(self, val):
        try:
            self.audio_target_rate = int(str(val).strip())
            self._log(f"Taxa de áudio alvo alterada para {self.audio_target_rate} Hz")
        except Exception:
            pass

    def _on_remux_toggle(self):
        self.remux_mp4 = bool(self.remux_var.get())
        self._log(f"Remux para MP4 {'ativado' if self.remux_mp4 else 'desativado'}")

    def _on_mic_toggle(self):
        self.mic_enabled = bool(self.mic_var.get())
        self._log(f"Microfone {'ativado' if self.mic_enabled else 'desativado'}")

    def _on_plugins_toggle(self):
        self.plugins_enabled = bool(self.plugins_enabled_var.get())
        self._log(f"Plugins {'ativados' if self.plugins_enabled else 'desativados'}")
        if self.plugins_enabled:
            self._load_plugins()
        else:
            self.plugin_objects.clear()
            self._refresh_plugins_list()

    def _region_to_str(self, region):
        if not region:
            return "Tela inteira"
        try:
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            return f"({x},{y}) {w}x{h} [Monitor {region['monitor']}]"
        except Exception:
            return "Tela inteira"

    def _select_region(self):
        self.region_win = tk.Toplevel(self.root)
        self.region_win.attributes("-fullscreen", True)
        self.region_win.attributes("-alpha", 0.3)
        self.region_win.attributes("-topmost", True)
        self.region_win.configure(bg="gray")

        self.region_canvas = tk.Canvas(self.region_win, highlightthickness=0)
        self.region_canvas.pack(fill=tk.BOTH, expand=True)

        self.region_win.bind("<Escape>", lambda e: self.region_win.destroy())
        self.region_win.bind("<ButtonPress-1>", self._on_region_press)
        self.region_win.bind("<B1-Motion>", self._on_region_drag)
        self.region_win.bind("<ButtonRelease-1>", self._on_region_release)

        if CTK_AVAILABLE:
            cancel_btn = ctk.CTkButton(self.region_win, text="Cancelar", command=self.region_win.destroy)
        else:
            cancel_btn = tk.Button(self.region_win, text="Cancelar", command=self.region_win.destroy)
        cancel_btn.place(relx=0.5, rely=0.95, anchor="center")

        self.region_start = None
        self.region_rect = None

    def _on_region_press(self, event):
        self.region_start = (event.x, event.y)

    def _on_region_drag(self, event):
        if self.region_start:
            if self.region_rect:
                self.region_canvas.delete(self.region_rect)
            x1, y1 = self.region_start
            x2, y2 = event.x, event.y
            self.region_rect = self.region_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def _on_region_release(self, event):
        if self.region_start:
            x1, y1 = self.region_start
            x2, y2 = event.x, event.y
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)

            # Dimensões pares ajudam encoders H.264/H.265
            w -= w % 2
            h -= h % 2

            if w > 10 and h > 10:
                self.region = {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "monitor": self.monitor_index,
                }
                self._set_entry(self.region_entry, self._region_to_str(self.region))
                self._log(f"Região selecionada: {self.region}")

        self.region_win.destroy()

    # ========== Monitor e áudio ==========
    def _get_monitor_size(self, idx):
        if not DXCAM_AVAILABLE:
            return 1920, 1080

        try:
            info = dxcam.output_info()
            if info and isinstance(info[0], dict):
                mon = info[idx]
                return int(mon["width"]), int(mon["height"])
        except Exception:
            pass

        try:
            cam = dxcam.create(output_idx=idx, output_color="BGR")
            cam.start()
            time.sleep(0.2)
            frame = cam.get_latest_frame()
            cam.stop()
            if frame is not None:
                h, w = frame.shape[:2]
                return w, h
        except Exception as e:
            self._log(f"Erro ao obter resolução do monitor {idx}: {e}", "error")

        return 1920, 1080

    def _update_monitor_list(self):
        names = []

        if DXCAM_AVAILABLE:
            try:
                monitors = dxcam.output_info()
                if monitors and isinstance(monitors[0], str):
                    names = [f"{i}: {name}" for i, name in enumerate(monitors)]
                else:
                    names = [
                        f"{i}: {m.get('name', 'Monitor')} ({m.get('width', '?')}x{m.get('height', '?')})"
                        for i, m in enumerate(monitors)
                    ]
            except Exception as e:
                self._log(f"Erro ao listar monitores: {e}", "error")

        if not names:
            names = ["0: Monitor padrão"]

        self._set_combobox_values(self.monitor_combo, names)

        if self.monitor_index is not None and 0 <= self.monitor_index < len(names):
            self.monitor_combo.set(names[self.monitor_index])
        else:
            self.monitor_combo.set(names[0])
            self.monitor_index = 0

        self._log(f"Monitores detectados: {len(names)}")

    def _on_monitor_change(self, val):
        try:
            idx = int(str(val).split(":")[0].strip())
            self.monitor_index = idx

            if self.region and self.region.get("monitor") != idx:
                self.region = None
                self._set_entry(self.region_entry, "Tela inteira")

            self._log(f"Monitor alterado para {idx}")
        except Exception:
            pass

    def _update_audio_devices(self):
        current_loopback = self.loopback_dev_combo.get() if hasattr(self, "loopback_dev_combo") else "Padrão"
        current_mic = self.mic_dev_combo.get() if hasattr(self, "mic_dev_combo") else "Padrão"

        devices = {"loopback": [], "mic": []}

        if AUDIO_AVAILABLE and pyaudio is not None:
            try:
                p = pyaudio.PyAudio()
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get("is_loopback", False):
                        devices["loopback"].append(f"{i}: {dev.get('name', 'Dispositivo')}")
                    elif dev.get("maxInputChannels", 0) > 0:
                        devices["mic"].append(f"{i}: {dev.get('name', 'Dispositivo')}")
                p.terminate()
            except Exception as e:
                self._log(f"Erro ao listar dispositivos de áudio: {e}", "error")

        if not devices["loopback"]:
            devices["loopback"] = ["Padrão"]
        if not devices["mic"]:
            devices["mic"] = ["Padrão"]

        self._set_combobox_values(self.loopback_dev_combo, devices["loopback"])
        self._set_combobox_values(self.mic_dev_combo, devices["mic"])

        if current_loopback in devices["loopback"]:
            self.loopback_dev_combo.set(current_loopback)
        else:
            self.loopback_dev_combo.set("Padrão")

        if current_mic in devices["mic"]:
            self.mic_dev_combo.set(current_mic)
        else:
            self.mic_dev_combo.set("Padrão")

        self._log("Dispositivos de áudio atualizados")

    def _on_loopback_dev_change(self, val):
        self.audio_device_loopback = val
        self._log(f"Loopback alterado para {val}")

    def _on_mic_dev_change(self, val):
        self.audio_device_mic = val
        self._log(f"Microfone alterado para {val}")

    def _browse_overlay_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.bmp")])
        if path:
            self.overlay_image = path
            self.overlay_image_entry.delete(0, tk.END)
            self.overlay_image_entry.insert(0, path)
            self._log(f"Overlay imagem carregada: {path}")

    # ========== Hotkeys ==========
    def _apply_hotkeys(self, silent=False):
        if not KEYBOARD_AVAILABLE:
            if not silent:
                self._show_info("Hotkeys", "Biblioteca 'keyboard' indisponível. Instale com: pip install keyboard")
            return

        for key in self.hotkeys_registered:
            try:
                keyboard.remove_hotkey(key)
            except Exception:
                pass
        self.hotkeys_registered.clear()

        start = self.start_hk_entry.get().strip()
        pause = self.pause_hk_entry.get().strip()
        mute = self.mute_hk_entry.get().strip()

        if start:
            self._add_hotkey(start, self.toggle_recording)
        if pause:
            self._add_hotkey(pause, self.toggle_pause)
        if mute and self.mic_enabled:
            self._add_hotkey(mute, self.toggle_mute_mic)

        self.start_hotkey = start
        self.pause_hotkey = pause
        self.mute_hotkey = mute

        self._log(f"Atalhos aplicados: start={start}, pause={pause}, mute={mute}")
        if not silent:
            self._show_info("Atalhos", "Atalhos aplicados!")

    def _add_hotkey(self, key, func):
        if not KEYBOARD_AVAILABLE or not key:
            return False
        try:
            try:
                keyboard.add_hotkey(key, func, suppress=False)
            except TypeError:
                keyboard.add_hotkey(key, func)
            self.hotkeys_registered.append(key)
            return True
        except Exception as e:
            self._log(f"Erro ao registrar hotkey {key}: {e}", "error")
            return False

    def _register_hotkeys(self):
        self._apply_hotkeys(silent=True)

    # ========== Plugins ==========
    def _load_plugins(self):
        self.plugin_objects.clear()

        enabled = self.plugins_enabled_var.get() if hasattr(self, "plugins_enabled_var") else self.plugins_enabled
        if not enabled:
            self._refresh_plugins_list()
            self._log("Plugins desabilitados")
            return

        if not os.path.exists(PLUGINS_DIR):
            self._refresh_plugins_list()
            return

        for filename in os.listdir(PLUGINS_DIR):
            if filename.endswith(".py") and not filename.startswith("__"):
                try:
                    path = os.path.join(PLUGINS_DIR, filename)
                    spec = importlib.util.spec_from_file_location(filename[:-3], path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "Plugin"):
                        plugin = module.Plugin()
                        self.plugin_objects.append(plugin)

                        on_load = getattr(plugin, "on_load", None)
                        if callable(on_load):
                            on_load()

                        self._log(f"Plugin carregado: {filename}")
                except Exception as e:
                    self._log(f"Erro ao carregar plugin {filename}: {e}", "error")

        self._refresh_plugins_list()

    def _refresh_plugins_list(self):
        if not hasattr(self, "plugins_listbox"):
            return
        self.plugins_listbox.delete(0, tk.END)
        for p in self.plugin_objects:
            self.plugins_listbox.insert(tk.END, p.__class__.__name__)

    def _apply_plugins(self, frame):
        enabled = self.plugins_enabled_var.get() if hasattr(self, "plugins_enabled_var") else self.plugins_enabled
        if not enabled:
            return frame

        for plugin in self.plugin_objects:
            try:
                new_frame = plugin.process_frame(frame)
                if new_frame is not None:
                    frame = new_frame
            except Exception as e:
                self._log(f"Erro plugin {plugin.__class__.__name__}: {e}", "error")
        return frame

    def _notify_plugins(self, event_name):
        enabled = self.plugins_enabled_var.get() if hasattr(self, "plugins_enabled_var") else self.plugins_enabled
        if not enabled:
            return

        for plugin in self.plugin_objects:
            try:
                method = getattr(plugin, event_name, None)
                if callable(method):
                    method()
            except Exception as e:
                self._log(f"Erro em plugin {plugin.__class__.__name__}.{event_name}: {e}", "error")

    # ========== Settings ==========
    def _save_current_settings(self):
        try:
            fps = int(self.fps_combo.get())
        except Exception:
            fps = self.fps

        self.settings.update(
            {
                "output_dir": self.dir_entry.get().strip(),
                "fps": fps,
                "mic_enabled": bool(self.mic_var.get()),
                "monitor_index": self.monitor_index,
                "region": self.region,
                "resize_resolution": list(self.resize_resolution) if self.resize_resolution else None,
                "video_codec": self.codec_combo.get(),
                "bitrate": self.audio_bitrate_entry.get().strip(),
                "video_bitrate": self.video_bitrate_entry.get().strip(),
                "start_hotkey": self.start_hk_entry.get().strip(),
                "pause_hotkey": self.pause_hk_entry.get().strip(),
                "mute_hotkey": self.mute_hk_entry.get().strip(),
                "schedule_start": self.sched_start_entry.get().strip(),
                "schedule_stop": self.sched_stop_entry.get().strip(),
                "show_indicator": bool(self.indicator_var.get()),
                "game_mode": bool(self.game_mode_var.get()),
                "low_latency": bool(self.low_latency_var.get()),
                "stream_enabled": bool(self.stream_enabled_var.get()),
                "stream_url": self.stream_url_entry.get().strip(),
                "stream_key": self.stream_key_entry.get().strip(),
                "stream_bitrate": self.stream_bitrate_entry.get().strip(),
                "stream_resolution": self.stream_res_entry.get().strip(),
                "overlay_text": self.overlay_text_entry.get(),
                "overlay_image": self.overlay_image_entry.get().strip(),
                "overlay_pos": self.overlay_pos_combo.get(),
                "audio_device_loopback": self.audio_device_loopback,
                "audio_device_mic": self.audio_device_mic,
                "plugins_enabled": bool(self.plugins_enabled_var.get()),
                "preview_size": list(self.preview_size),
                "schedule_active": self.schedule_active,
                "remux_mp4": bool(self.remux_var.get()),
                "encoder_preset": self.encoder_preset_combo.get(),
                "audio_sample_rate": int(self.audio_rate_combo.get() or 48000),
            }
        )

        self.settings = _normalize_settings(self.settings)
        save_settings(self.settings)

        # Atualiza atributos importantes
        self.fps = int(self.settings["fps"])
        self.video_codec = self.settings["video_codec"]
        self.bitrate = self.settings["bitrate"]
        self.video_bitrate = self.settings["video_bitrate"]
        self.start_hotkey = self.settings["start_hotkey"]
        self.pause_hotkey = self.settings["pause_hotkey"]
        self.mute_hotkey = self.settings["mute_hotkey"]
        self.schedule_start = self.settings["schedule_start"]
        self.schedule_stop = self.settings["schedule_stop"]
        self.show_indicator = self.settings["show_indicator"]
        self.game_mode = self.settings["game_mode"]
        self.low_latency = self.settings["low_latency"]
        self.stream_enabled = self.settings["stream_enabled"]
        self.stream_url = self.settings["stream_url"]
        self.stream_key = self.settings["stream_key"]
        self.stream_bitrate = self.settings["stream_bitrate"]
        self.stream_resolution = self.settings["stream_resolution"]
        self.overlay_text = self.settings["overlay_text"]
        self.overlay_image = self.settings["overlay_image"]
        self.overlay_pos = self.settings["overlay_pos"]
        self.plugins_enabled = self.settings["plugins_enabled"]
        self.remux_mp4 = self.settings["remux_mp4"]
        self.encoder_preset = self.settings["encoder_preset"]
        self.audio_target_rate = int(self.settings["audio_sample_rate"])
        self.output_dir = self.settings["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self._log("Configurações salvas")
        self._show_info("Salvo", "Configurações salvas com sucesso!")

    # ========== Agendamento ==========
    def _next_schedule_datetimes(self, start_str, stop_str):
        t_start = datetime.strptime(start_str, "%H:%M").time()
        t_end = datetime.strptime(stop_str, "%H:%M").time()

        now = datetime.now()
        start_dt = datetime.combine(now.date(), t_start)
        end_dt = datetime.combine(now.date(), t_end)

        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        if start_dt <= now:
            start_dt += timedelta(days=1)
            end_dt += timedelta(days=1)

        return start_dt, end_dt

    def _toggle_schedule(self):
        if self.schedule_thread and self.schedule_thread.is_alive():
            self.schedule_stop_event.set()
            self.schedule_thread.join(timeout=2)
            self.schedule_thread = None
            self.sched_btn.configure(text="Ativar")
            self.schedule_active = False
            self._log("Agendamento cancelado")
            return

        start_str = self.sched_start_entry.get().strip()
        stop_str = self.sched_stop_entry.get().strip()

        if not start_str or not stop_str:
            self._show_error("Erro", "Defina horários de início e fim.")
            return

        try:
            start_dt, end_dt = self._next_schedule_datetimes(start_str, stop_str)
        except Exception:
            self._show_error("Erro", "Formato de hora inválido (HH:MM).")
            return

        self.schedule_stop_event.clear()
        self.schedule_thread = threading.Thread(target=self._schedule_worker, args=(start_dt, end_dt), daemon=True)
        self.schedule_thread.start()

        self.sched_btn.configure(text="Cancelar")
        self.schedule_active = True
        self.schedule_start = start_str
        self.schedule_stop = stop_str
        self._log(f"Agendamento ativado: início {start_dt}, fim {end_dt}")

    def _schedule_worker(self, start_dt, end_dt):
        while datetime.now() < start_dt and not self.schedule_stop_event.is_set():
            time.sleep(1)

        if self.schedule_stop_event.is_set():
            return

        self._ui(self.start_recording)

        while datetime.now() < end_dt and not self.schedule_stop_event.is_set():
            time.sleep(1)

        self._ui(self.stop_recording)
        self._ui(lambda: self.sched_btn.configure(text="Ativar"))
        self.schedule_active = False

    def _restore_schedule(self):
        if not self.schedule_start or not self.schedule_stop:
            return

        try:
            start_dt, end_dt = self._next_schedule_datetimes(self.schedule_start, self.schedule_stop)
            self.schedule_stop_event.clear()
            self.schedule_thread = threading.Thread(target=self._schedule_worker, args=(start_dt, end_dt), daemon=True)
            self.schedule_thread.start()
            self.sched_btn.configure(text="Cancelar")
            self.schedule_active = True
            self._log("Agendamento restaurado")
        except Exception:
            pass

    # ========== Áudio ==========
    def _compute_rms(self, data):
        if not data:
            return 0.0
        try:
            arr = np.frombuffer(data, dtype=np.int16)
            if arr.size == 0:
                return 0.0
            rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2))
            return min(rms / 32768.0 * 100.0, 100.0)
        except Exception:
            return 0.0

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.recording and not self.paused and self.audio_file:
            try:
                self.audio_file.write(in_data)
            except Exception:
                pass

        rms = self._compute_rms(in_data)
        with self.rms_lock:
            self.smoothed_loopback = 0.7 * self.smoothed_loopback + 0.3 * rms

        return (in_data, pyaudio.paContinue)

    def _mic_callback(self, in_data, frame_count, time_info, status):
        if self.recording and not self.paused and self.mic_file:
            if self.mic_muted:
                try:
                    silence = b"\x00" * (frame_count * 2)
                    self.mic_file.write(silence)
                except Exception:
                    pass
                with self.rms_lock:
                    self.smoothed_mic = 0.7 * self.smoothed_mic
            else:
                try:
                    self.mic_file.write(in_data)
                except Exception:
                    pass
                rms = self._compute_rms(in_data)
                with self.rms_lock:
                    self.smoothed_mic = 0.7 * self.smoothed_mic + 0.3 * rms
        else:
            if self.mic_muted:
                with self.rms_lock:
                    self.smoothed_mic = 0.7 * self.smoothed_mic
            else:
                rms = self._compute_rms(in_data)
                with self.rms_lock:
                    self.smoothed_mic = 0.7 * self.smoothed_mic + 0.3 * rms

        return (in_data, pyaudio.paContinue)

    # ========== Utilitários de vídeo/encoder ==========
    def _parse_resolution(self, text):
        try:
            parts = str(text).strip().lower().split("x")
            w, h = int(parts[0]), int(parts[1])
            if w > 0 and h > 0:
                w -= w % 2
                h -= h % 2
                return w, h
        except Exception:
            pass
        return None

    def _clean_bitrate(self, value):
        v = str(value or "").strip()
        if not v or v.lower() == "auto":
            return ""
        return v

    def _check_ffmpeg(self):
        try:
            if shutil.which("ffmpeg") is None:
                return False
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return True
        except Exception:
            return False

    def _ffmpeg_encoder_banner(self):
        if getattr(self, "_ffmpeg_encoder_cache", None) is not None:
            return self._ffmpeg_encoder_cache

        if not self.ffmpeg_available:
            self._ffmpeg_encoder_cache = ""
            return self._ffmpeg_encoder_cache

        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            self._ffmpeg_encoder_cache = proc.stdout or ""
        except Exception:
            self._ffmpeg_encoder_cache = ""

        return self._ffmpeg_encoder_cache

    def _has_ffmpeg_encoder(self, name):
        banner = self._ffmpeg_encoder_banner()
        return f" {name} " in banner or f"{name} " in banner

    def _pick_ffmpeg_encoder(self, codec):
        if not self.ffmpeg_available:
            return None

        codec = str(codec or "auto").lower()
        if codec == "copy":
            return None

        vb = self._clean_bitrate(self.video_bitrate)

        if codec == "h265":
            candidates = ["hevc_nvenc", "hevc_amf", "hevc_qsv", "libx265"]
        else:
            candidates = ["h264_nvenc", "h264_amf", "h264_qsv", "libx264"]

        for enc in candidates:
            if not self._has_ffmpeg_encoder(enc):
                continue

            if enc.endswith("nvenc"):
                args = ["-c:v", enc]
                if vb:
                    args += ["-rc", "vbr", "-b:v", vb]
                else:
                    args += ["-rc", "constqp", "-qp", "23"]
                args += ["-preset", "p4" if self.low_latency else "p5"]
                if self.low_latency:
                    args += ["-tune", "ll"]
                args += ["-pix_fmt", "yuv420p"]
                return args

            if enc.endswith("amf"):
                args = ["-c:v", enc, "-b:v", vb or "8000k", "-pix_fmt", "yuv420p"]
                return args

            if enc.endswith("qsv"):
                args = ["-c:v", enc, "-b:v", vb or "8000k", "-pix_fmt", "yuv420p"]
                return args

            if enc == "libx264":
                args = [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast" if self.low_latency else self.encoder_preset,
                ]
                if vb:
                    args += ["-b:v", vb]
                else:
                    args += ["-crf", "23"]
                args += ["-pix_fmt", "yuv420p"]
                if self.low_latency:
                    args += ["-tune", "zerolatency"]
                return args

            if enc == "libx265":
                args = [
                    "-c:v",
                    "libx265",
                    "-preset",
                    "ultrafast" if self.low_latency else "medium",
                ]
                if vb:
                    args += ["-b:v", vb]
                else:
                    args += ["-crf", "28"]
                args += ["-pix_fmt", "yuv420p"]
                return args

        return None

    def _pick_stream_encoder_args(self, bitrate):
        bitrate = self._clean_bitrate(bitrate) or "2000k"

        if self._has_ffmpeg_encoder("h264_nvenc"):
            args = [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-b:v",
                bitrate,
                "-pix_fmt",
                "yuv420p",
            ]
            if self.low_latency:
                args += ["-tune", "ll"]
            return args

        if self._has_ffmpeg_encoder("h264_amf"):
            return ["-c:v", "h264_amf", "-b:v", bitrate, "-pix_fmt", "yuv420p"]

        if self._has_ffmpeg_encoder("h264_qsv"):
            return ["-c:v", "h264_qsv", "-b:v", bitrate, "-pix_fmt", "yuv420p"]

        args = [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast" if self.low_latency else "veryfast",
            "-b:v",
            bitrate,
            "-pix_fmt",
            "yuv420p",
        ]
        if self.low_latency:
            args += ["-tune", "zerolatency"]
        return args

    def _create_video_writer(self, path, fps, size):
        # Fallback mais compatível: MJPG em AVI
        candidates = [
            ("MJPG", ".avi"),
            ("XVID", ".avi"),
        ]

        base_no_ext = os.path.splitext(path)[0]

        for codec, ext in candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_path = base_no_ext + ext
            writer = cv2.VideoWriter(test_path, fourcc, fps, size)
            if writer.isOpened():
                return writer, test_path
            writer.release()

        raise RuntimeError("Nenhum codec de vídeo disponível no fallback OpenCV.")

    def _raw_to_wav(self, raw_path, wav_path, rate, channels):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            "-i",
            raw_path,
            "-af",
            f"aresample={self.audio_target_rate},aformat=sample_fmts=s16:channel_layouts=stereo",
            "-c:a",
            "pcm_s16le",
            wav_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    # ========== Overlay cache ==========
    def _get_overlay_asset(self, out_w, out_h):
        if not self.overlay_image:
            return None

        try:
            st = os.stat(self.overlay_image)
        except OSError:
            return None

        key = (self.overlay_image, st.st_mtime, out_w, out_h)
        if key in self._overlay_cache:
            return self._overlay_cache[key]

        img = cv2.imread(self.overlay_image, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        h_img, w_img = img.shape[:2]
        if h_img <= 0 or w_img <= 0:
            return None

        scale = min(out_w / float(w_img), out_h / float(h_img)) * 0.2
        new_w = max(1, int(w_img * scale))
        new_h = max(1, int(h_img * scale))

        if (new_w, new_h) != (w_img, h_img):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3].astype(np.float32) / 255.0
            bgr = img[:, :, :3]
        else:
            alpha = None
            bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        asset = (bgr, alpha, new_w, new_h)

        if len(self._overlay_cache) > 8:
            self._overlay_cache.clear()

        self._overlay_cache[key] = asset
        return asset

    # ========== Vídeo / Preview ==========
    def _video_loop(self):
        frame_interval = 1.0 / max(1, self.fps)
        next_frame_time = time.perf_counter()
        last_preview = 0.0

        while not self.stop_event.is_set() and self.recording:
            if self.paused:
                time.sleep(0.05)
                next_frame_time = time.perf_counter() + frame_interval
                continue

            frame = self.camera.get_latest_frame() if self.camera else None

            if frame is not None:
                # Região
                if self.region:
                    reg = self.region
                    if reg.get("monitor") == self.monitor_index:
                        x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
                        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                            frame = frame[y:y + h, x:x + w]

                # Resize
                if self.resize_resolution and (frame.shape[1], frame.shape[0]) != tuple(self.resize_resolution):
                    frame = cv2.resize(frame, tuple(self.resize_resolution))

                needs_compose = bool(self.overlay_text or self.overlay_image or self.plugin_objects)

                if needs_compose:
                    out_frame = frame.copy()
                else:
                    out_frame = frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)

                # Overlay de texto
                if self.overlay_text:
                    cv2.putText(
                        out_frame,
                        self.overlay_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Overlay de imagem com cache
                if self.overlay_image:
                    asset = self._get_overlay_asset(out_frame.shape[1], out_frame.shape[0])
                    if asset:
                        img, alpha, new_w, new_h = asset
                        h_frame, w_frame = out_frame.shape[:2]

                        if self.overlay_pos == "top-left":
                            x0, y0 = 0, 0
                        elif self.overlay_pos == "top-right":
                            x0 = max(0, w_frame - new_w)
                            y0 = 0
                        elif self.overlay_pos == "bottom-left":
                            x0 = 0
                            y0 = max(0, h_frame - new_h)
                        else:  # bottom-right
                            x0 = max(0, w_frame - new_w)
                            y0 = max(0, h_frame - new_h)

                        if x0 >= 0 and y0 >= 0 and x0 + new_w <= w_frame and y0 + new_h <= h_frame:
                            roi = out_frame[y0:y0 + new_h, x0:x0 + new_w]

                            if alpha is not None:
                                a = alpha[..., np.newaxis]
                                out_frame[y0:y0 + new_h, x0:x0 + new_w] = (
                                    roi.astype(np.float32) * (1.0 - a) + img.astype(np.float32) * a
                                ).astype(np.uint8)
                            else:
                                out_frame[y0:y0 + new_h, x0:x0 + new_w] = img

                # Plugins
                out_frame = self._apply_plugins(out_frame)

                # Gravação
                if self.video_writer:
                    try:
                        self.video_writer.write(out_frame)
                    except Exception as e:
                        self._log(f"Erro ao escrever frame: {e}", "error")
                        break

                # Streaming
                if self.stream_process and self.stream_process.poll() is None:
                    try:
                        self.stream_process.stdin.write(out_frame.tobytes())
                    except (BrokenPipeError, OSError) as e:
                        self._log(f"Erro no stream: {e}", "error")
                        self.stream_process = None

                # Preview
                now = time.perf_counter()
                if now - last_preview > 0.2:
                    preview_frame = out_frame.copy()
                    self._ui(self._update_preview, preview_frame)
                    last_preview = now

            now = time.perf_counter()
            sleep_time = next_frame_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            next_frame_time += frame_interval
            if next_frame_time < now:
                next_frame_time = now + frame_interval

    def _update_preview(self, frame):
        try:
            if frame is None:
                return

            self.last_preview_frame = frame

            w, h = self.preview_size
            if w <= 1 or h <= 1:
                return

            preview_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            self.preview_img = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_img)
        except Exception:
            pass

    def _on_preview_resize(self, event):
        if event.width > 20 and event.height > 20:
            self.preview_size = (event.width, event.height)
            if self.last_preview_frame is not None:
                self._update_preview(self.last_preview_frame)

    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        if self.paused:
            return int(self._elapsed_before_pause)
        return int(time.perf_counter() - self.start_time)

    def _update_vu_meters(self):
        try:
            def set_bar(bar, value):
                if CTK_AVAILABLE:
                    bar.set(value / 100.0)
                else:
                    bar["value"] = value

            with self.rms_lock:
                set_bar(self.loopback_vu, self.smoothed_loopback)
                set_bar(self.mic_vu, self.smoothed_mic)

            self.root.after(100, self._update_vu_meters)
        except Exception:
            return

    def _update_time_label(self):
        try:
            if not self.recording:
                return

            elapsed = self.get_elapsed_time()
            mins, secs = divmod(elapsed, 60)
            self.label_time.configure(text=f"{mins:02d}:{secs:02d}")
            self.root.after(200, self._update_time_label)
        except Exception:
            return

    # ========== UI de estado ==========
    def _ui_set_recording_started(self):
        try:
            if CTK_AVAILABLE:
                self.btn.configure(text="⏹ PARAR", fg_color="red", text_color="white")
            else:
                self.btn.configure(text="⏹ PARAR", bg="red", fg="white")

            self.pause_btn.configure(state="normal", text="⏸ PAUSAR")

            if self.mic_btn:
                self.mic_btn.configure(state="normal", text="🎤 Ligado")
        except Exception:
            pass

    def _ui_set_recording_stopped(self):
        try:
            if CTK_AVAILABLE:
                self.btn.configure(text="▶ GRAVAR", fg_color="transparent", text_color=("black", "white"))
            else:
                try:
                    bg = self.root.cget("bg")
                except Exception:
                    bg = "SystemButtonFace"
                self.btn.configure(text="▶ GRAVAR", bg=bg, fg="black")

            self.pause_btn.configure(state="disabled", text="⏸ PAUSAR")

            if self.mic_btn:
                self.mic_btn.configure(state="disabled", text="🎤 Ligado")

            self.label_time.configure(text="00:00")
        except Exception:
            pass

    # ========== Start / Stop ==========
    def start_recording(self):
        if self.recording:
            return

        if not DXCAM_AVAILABLE:
            self._show_error("Erro", "dxcam não está disponível. Instale com: pip install dxcam")
            return

        if self.video_codec in ("auto", "h264", "h265") and not self.ffmpeg_available:
            self.video_codec = "copy"
            self._log("FFmpeg indisponível; codec alterado para copy.", "warning")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"obsx_{timestamp}"

        if self.ffmpeg_available:
            output_ext = ".mp4" if self.remux_mp4 else ".mkv"
            temp_video = os.path.join(self.output_dir, f"{base}_video.mkv")
        else:
            output_ext = ".avi"
            temp_video = os.path.join(self.output_dir, f"{base}_video.avi")

        self.output_path = os.path.join(self.output_dir, f"{base}{output_ext}")
        self.video_path = temp_video

        # Tamanho de saída
        if self.resize_resolution:
            out_w, out_h = self.resize_resolution
        elif self.region:
            out_w, out_h = self.region["w"], self.region["h"]
        else:
            out_w, out_h = self._get_monitor_size(self.monitor_index)

        out_w = int(out_w)
        out_h = int(out_h)
        out_w -= out_w % 2
        out_h -= out_h % 2

        if out_w <= 0 or out_h <= 0:
            self._show_error("Erro", "Resolução de captura inválida.")
            return

        self.current_output_size = (out_w, out_h)

        # Writer
        self.video_writer = None

        if self.ffmpeg_available and self.video_codec in ("auto", "h264", "h265"):
            encoder_args = self._pick_ffmpeg_encoder(self.video_codec)
            if encoder_args:
                try:
                    writer = FFmpegWriter(temp_video, self.fps, (out_w, out_h), encoder_args)
                    if writer.isOpened():
                        self.video_writer = writer
                        self.video_path = temp_video
                    else:
                        writer.release()
                        raise RuntimeError("FFmpeg writer não abriu")
                except Exception as e:
                    self._log(f"FFmpeg writer falhou, usando fallback OpenCV: {e}", "error")
                    self.video_writer = None

        if self.video_writer is None:
            fallback_path = os.path.join(self.output_dir, f"{base}_video.avi")
            try:
                self.video_writer, self.video_path = self._create_video_writer(
                    fallback_path,
                    self.fps,
                    (out_w, out_h),
                )
            except Exception as e:
                self._show_error("Erro", f"Falha ao criar VideoWriter: {e}")
                self._log(f"Erro VideoWriter: {e}", "error")
                return

        # Câmera
        try:
            self.camera = dxcam.create(output_color="BGR", output_idx=self.monitor_index)
            self.camera.start(target_fps=self.fps, video_mode=self.game_mode)
        except Exception as e:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None
            self.camera = None
            self._show_error("Erro", f"Falha ao iniciar captura: {e}")
            self._log(f"Erro captura: {e}", "error")
            return

        self.recording = True
        self.paused = False
        self.mic_muted = False
        self.stop_event.clear()
        self.start_time = time.perf_counter()
        self._elapsed_before_pause = 0.0
        self.smoothed_loopback = 0.0
        self.smoothed_mic = 0.0

        # Áudio
        self.audio_file = None
        self.audio_file_path = None
        self.mic_file = None
        self.mic_file_path = None
        self.audio_stream = None
        self.mic_stream = None

        if AUDIO_AVAILABLE and pyaudio is not None:
            try:
                self.p_audio = pyaudio.PyAudio()
            except Exception as e:
                self._log(f"Erro ao iniciar PyAudio: {e}", "error")
                self.p_audio = None

            # Loopback
            if self.p_audio:
                try:
                    dev_idx = None
                    if self.audio_device_loopback and str(self.audio_device_loopback) != "Padrão":
                        try:
                            dev_idx = int(str(self.audio_device_loopback).split(":")[0])
                        except Exception:
                            dev_idx = None

                    loopback = None
                    if dev_idx is None:
                        if hasattr(self.p_audio, "get_default_wasapi_loopback"):
                            loopback = self.p_audio.get_default_wasapi_loopback()
                    else:
                        loopback = self.p_audio.get_device_info_by_index(dev_idx)

                    if loopback:
                        self.audio_channels = int(loopback.get("maxInputChannels", 2)) or 2
                        self.audio_rate = int(loopback.get("defaultSampleRate", 48000)) or 48000

                        fd, raw_path = tempfile.mkstemp(suffix=".raw", prefix="loopback_")
                        os.close(fd)
                        self.audio_file_path = raw_path
                        self.audio_file = open(raw_path, "wb")

                        self.audio_stream = self.p_audio.open(
                            format=pyaudio.paInt16,
                            channels=self.audio_channels,
                            rate=self.audio_rate,
                            input=True,
                            input_device_index=loopback.get("index"),
                            stream_callback=self._audio_callback,
                        )
                        self.audio_stream.start_stream()
                        self._log(f"Loopback ativado (canais {self.audio_channels}, taxa {self.audio_rate})")
                    else:
                        self._log("Nenhum dispositivo loopback disponível.", "warning")
                except Exception as e:
                    self._log(f"Erro loopback: {e}", "error")

            # Microfone
            if self.mic_enabled and self.p_audio:
                try:
                    dev_idx = None
                    if self.audio_device_mic and str(self.audio_device_mic) != "Padrão":
                        try:
                            dev_idx = int(str(self.audio_device_mic).split(":")[0])
                        except Exception:
                            dev_idx = None

                    fd, raw_path = tempfile.mkstemp(suffix=".raw", prefix="mic_")
                    os.close(fd)
                    self.mic_file_path = raw_path
                    self.mic_file = open(raw_path, "wb")

                    self.mic_stream = self.p_audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        input_device_index=dev_idx,
                        stream_callback=self._mic_callback,
                    )
                    self.mic_stream.start_stream()
                    self._log("Microfone ativado")
                except Exception as e:
                    self._log(f"Erro microfone: {e}", "error")

        # Streaming
        stream_enabled = self.stream_enabled_var.get() if hasattr(self, "stream_enabled_var") else self.stream_enabled
        if stream_enabled and self.ffmpeg_available:
            self._start_stream()

        # Thread de vídeo
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

        self._notify_plugins("on_start")
        self._create_indicator()
        self._ui(self._ui_set_recording_started)
        self._ui(self._update_time_label)

        self._log(f"Gravação iniciada. Saída: {self.output_path}")
        print(f"🎥 GRAVANDO... Saída: {self.output_path}")

    def _start_stream(self):
        if not self.ffmpeg_available:
            self._log("FFmpeg não disponível para streaming", "error")
            return

        url = self.stream_url_entry.get().strip()
        key = self.stream_key_entry.get().strip()

        if not url or not key:
            self._log("URL ou chave de stream vazia", "error")
            return

        rtmp_url = f"{url.rstrip('/')}/{key}"

        in_w, in_h = self.current_output_size if self.current_output_size else (1920, 1080)
        stream_size = self._parse_resolution(self.stream_res_entry.get())
        bitrate = self.stream_bitrate_entry.get().strip() or "2000k"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{in_w}x{in_h}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
        ]

        if stream_size and stream_size != (in_w, in_h):
            cmd += ["-vf", f"scale={stream_size[0]}:{stream_size[1]}"]

        cmd += self._pick_stream_encoder_args(bitrate)
        cmd += [
            "-g",
            str(max(1, self.fps * 2)),
            "-f",
            "flv",
            rtmp_url,
        ]

        popen_kwargs = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            self.stream_process = subprocess.Popen(cmd, **popen_kwargs)
            self._log(f"Streaming iniciado para {rtmp_url}")
            self._log("Aviso: neste protótipo, o stream envia apenas vídeo.", "warning")
        except Exception as e:
            self._log(f"Erro ao iniciar stream: {e}", "error")
            self.stream_process = None

    def _test_stream(self):
        if not self.ffmpeg_available:
            self._show_error("Stream", "FFmpeg não encontrado.")
            return

        url = self.stream_url_entry.get().strip()
        key = self.stream_key_entry.get().strip()

        if not url or not key:
            self._show_error("Stream", "Preencha URL e chave de stream.")
            return

        self._show_info("Stream", "Configuração de stream válida. Inicie a gravação para transmitir.")

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        self.stop_event.set()
        self._destroy_indicator()

        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=3.0)

        self._notify_plugins("on_stop")

        # Stream
        if self.stream_process and self.stream_process.poll() is None:
            try:
                self.stream_process.stdin.close()
                self.stream_process.wait(timeout=5)
            except Exception:
                try:
                    self.stream_process.kill()
                except Exception:
                    pass
            self.stream_process = None
            self._log("Stream encerrado")

        # Áudio
        for stream in (self.audio_stream, self.mic_stream):
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

        self.audio_stream = None
        self.mic_stream = None

        if self.p_audio:
            try:
                self.p_audio.terminate()
            except Exception:
                pass
            self.p_audio = None

        for f in (self.audio_file, self.mic_file):
            if f:
                try:
                    f.close()
                except Exception:
                    pass

        self.audio_file = None
        self.mic_file = None

        # Câmera
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None

        # Writer
        if self.video_writer:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        video_ok = bool(self.video_path and os.path.exists(self.video_path) and os.path.getsize(self.video_path) > 0)
        audio_ok = bool(
            self.audio_file_path
            and os.path.exists(self.audio_file_path)
            and os.path.getsize(self.audio_file_path) > 0
        )
        mic_ok = bool(
            self.mic_file_path
            and os.path.exists(self.mic_file_path)
            and os.path.getsize(self.mic_file_path) > 0
        )

        if video_ok and (audio_ok or mic_ok) and self.ffmpeg_available:
            self._merge_with_ffmpeg()
        elif video_ok:
            self._finalize_video_only()
        else:
            self._log("Nenhum vídeo gerado.", "error")

        self._ui(self._ui_set_recording_stopped)
        self._log("Gravação finalizada.")

    # ========== Pause / Mute ==========
    def toggle_pause(self):
        if not self.recording:
            return

        with self.pause_lock:
            if self.paused:
                self.paused = False
                self.start_time = time.perf_counter() - self._elapsed_before_pause
                self._ui(lambda: self.pause_btn.configure(text="⏸ PAUSAR"))
                self._log("Gravação retomada")
            else:
                self.paused = True
                self._elapsed_before_pause = time.perf_counter() - self.start_time
                self._ui(lambda: self.pause_btn.configure(text="▶ RETOMAR"))
                self._log("Gravação pausada")

    def toggle_mute_mic(self):
        if not self.recording or not self.mic_enabled:
            return

        self.mic_muted = not self.mic_muted

        def _update():
            if self.mic_btn:
                self.mic_btn.configure(text="🎤 Mudo" if self.mic_muted else "🎤 Ligado")

        self._ui(_update)
        self._log(f"Microfone {'mutado' if self.mic_muted else 'reativado'}")

    # ========== FFmpeg merge/finalização ==========
    def _merge_with_ffmpeg(self):
        audio_inputs = []

        if self.audio_file_path and os.path.exists(self.audio_file_path) and os.path.getsize(self.audio_file_path) > 0:
            wav_loop = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            try:
                self._raw_to_wav(self.audio_file_path, wav_loop, self.audio_rate, self.audio_channels)
                audio_inputs.append(("loopback", wav_loop))
            except Exception as e:
                self._log(f"Erro ao converter loopback: {e}", "error")

        if self.mic_file_path and os.path.exists(self.mic_file_path) and os.path.getsize(self.mic_file_path) > 0:
            wav_mic = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            try:
                self._raw_to_wav(self.mic_file_path, wav_mic, 44100, 1)
                audio_inputs.append(("mic", wav_mic))
            except Exception as e:
                self._log(f"Erro ao converter microfone: {e}", "error")

        if not audio_inputs:
            self._finalize_video_only()
            return

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            self.video_path,
        ]

        for _, wav in audio_inputs:
            cmd += ["-i", wav]

        if len(audio_inputs) == 1:
            cmd.extend([
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:a",
                "aac",
                "-b:a",
                self.bitrate or "192k",
                "-ar",
                str(self.audio_target_rate),
            ])
        else:
            cmd.extend([
                "-filter_complex",
                "[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=0[a]",
                "-map",
                "0:v",
                "-map",
                "[a]",
                "-c:a",
                "aac",
                "-b:a",
                self.bitrate or "192k",
                "-ar",
                str(self.audio_target_rate),
            ])

        video_copy = self.video_path.lower().endswith((".mkv", ".mp4"))

        if video_copy:
            cmd.extend(["-c:v", "copy"])
            if self.output_path.lower().endswith(".mp4") and self.video_codec == "h265":
                cmd.extend(["-tag:v", "hvc1"])
        else:
            final_encoder = self._pick_ffmpeg_encoder(self.video_codec if self.video_codec != "copy" else "auto")
            if final_encoder:
                cmd.extend(final_encoder)
            else:
                cmd.extend([
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                ])

        if self.output_path.lower().endswith(".mp4"):
            cmd.extend(["-movflags", "+faststart"])

        cmd.extend(["-shortest", self.output_path])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self._log(f"Merge concluído: {self.output_path}")

            try:
                if os.path.exists(self.video_path):
                    os.remove(self.video_path)
                for _, wav in audio_inputs:
                    if os.path.exists(wav):
                        os.remove(wav)
                if self.audio_file_path and os.path.exists(self.audio_file_path):
                    os.remove(self.audio_file_path)
                if self.mic_file_path and os.path.exists(self.mic_file_path):
                    os.remove(self.mic_file_path)
            except Exception:
                pass

        except subprocess.CalledProcessError as e:
            stderr = ""
            try:
                stderr = e.stderr.decode(errors="ignore")
            except Exception:
                pass
            self._log(f"Erro FFmpeg: {stderr}", "error")

    def _finalize_video_only(self):
        if not self.ffmpeg_available:
            final = self.video_path
            try:
                if self.output_path and final != self.output_path:
                    os.replace(final, self.output_path)
                    final = self.output_path
            except Exception:
                pass
            self._log(f"Vídeo salvo como: {final}")
            return

        if self.video_path == self.output_path:
            self._log(f"Vídeo final: {self.output_path}")
            return

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            self.video_path,
        ]

        video_copy = self.video_path.lower().endswith((".mkv", ".mp4"))

        if video_copy:
            cmd.extend(["-c:v", "copy"])
            if self.output_path.lower().endswith(".mp4") and self.video_codec == "h265":
                cmd.extend(["-tag:v", "hvc1"])
        else:
            final_encoder = self._pick_ffmpeg_encoder(self.video_codec if self.video_codec != "copy" else "auto")
            if final_encoder:
                cmd.extend(final_encoder)
            else:
                cmd.extend([
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                ])

        if self.output_path.lower().endswith(".mp4"):
            cmd.extend(["-movflags", "+faststart"])

        cmd.append(self.output_path)

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.exists(self.video_path):
                os.remove(self.video_path)
            self._log(f"Vídeo final (sem áudio): {self.output_path}")
        except Exception as e:
            self._log(f"Erro ao converter vídeo: {e}", "error")

    # ========== Indicator ==========
    def _create_indicator(self):
        indicator_enabled = self.indicator_var.get() if hasattr(self, "indicator_var") else self.show_indicator
        if not indicator_enabled:
            return

        try:
            self.indicator = tk.Toplevel(self.root)
            self.indicator.overrideredirect(True)
            self.indicator.geometry("24x24+10+10")
            self.indicator.attributes("-topmost", True)

            try:
                self.indicator.attributes("-transparentcolor", "white")
                bg = "white"
            except Exception:
                bg = "black"

            self.indicator.configure(bg=bg)

            canvas = tk.Canvas(self.indicator, width=24, height=24, bg=bg, highlightthickness=0)
            canvas.pack()
            canvas.create_oval(2, 2, 22, 22, fill="red", outline="")
        except Exception:
            self.indicator = None

    def _destroy_indicator(self):
        try:
            if hasattr(self, "indicator") and self.indicator:
                self.indicator.destroy()
                self.indicator = None
        except Exception:
            self.indicator = None

    # ========== Cleanup ==========
    def _cleanup(self):
        if getattr(self, "_cleaned", False):
            return
        self._cleaned = True

        if KEYBOARD_AVAILABLE:
            try:
                for key in self.hotkeys_registered:
                    keyboard.remove_hotkey(key)
            except Exception:
                pass
        self.hotkeys_registered.clear()

        try:
            if self.recording:
                self.stop_recording()
        except Exception:
            pass

        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None

        if self.video_writer:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        if self.stream_process and self.stream_process.poll() is None:
            try:
                self.stream_process.kill()
            except Exception:
                pass
            self.stream_process = None

        for stream in (self.audio_stream, self.mic_stream):
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

        self.audio_stream = None
        self.mic_stream = None

        if self.p_audio:
            try:
                self.p_audio.terminate()
            except Exception:
                pass
            self.p_audio = None

        for f in (self.audio_file, self.mic_file):
            if f:
                try:
                    f.close()
                except Exception:
                    pass

        self.audio_file = None
        self.mic_file = None

        self._destroy_indicator()

    def _signal_handler(self, sig, frame):
        self._cleanup()
        sys.exit(0)

    def _on_close(self):
        self._cleanup()
        try:
            self.root.destroy()
        except Exception:
            pass

    # ========== Toggle principal ==========
    def toggle_recording(self):
        with self._lock:
            if self._toggling:
                return
            self._toggling = True

        def _run():
            try:
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            finally:
                with self._lock:
                    self._toggling = False

        threading.Thread(target=_run, daemon=True).start()

    def run(self):
        print("✅ OBSX pronto!")
        print("📌 F12 = Iniciar/Parar, F11 = Pausar")
        if not self.ffmpeg_available:
            print("⚠️ FFmpeg não encontrado – instale para melhor codificação, merge e streaming.")
        self._log("Aplicação em execução")
        self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBSX - Gravador profissional leve")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--reencode", action="store_true")
    parser.add_argument("--output", type=str)
    parser.add_argument("--mic", action="store_true")
    parser.add_argument("--no-indicator", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Habilitar streaming")
    parser.add_argument("--stream-url", type=str)
    parser.add_argument("--stream-key", type=str)
    parser.add_argument("--codec", type=str)
    parser.add_argument("--video-bitrate", type=str)
    parser.add_argument("--audio-bitrate", type=str)

    args = parser.parse_args()

    kwargs = {}

    if args.fps is not None:
        kwargs["fps"] = args.fps
    if args.reencode:
        kwargs["reencode"] = True
    if args.output is not None:
        kwargs["output_dir"] = args.output
    if args.mic:
        kwargs["mic_enabled"] = True
    if args.no_indicator:
        kwargs["show_indicator"] = False
    if args.stream:
        kwargs["stream_enabled"] = True
    if args.stream_url is not None:
        kwargs["stream_url"] = args.stream_url
    if args.stream_key is not None:
        kwargs["stream_key"] = args.stream_key
    if args.codec is not None:
        kwargs["video_codec"] = args.codec
    if args.video_bitrate is not None:
        kwargs["video_bitrate"] = args.video_bitrate
    if args.audio_bitrate is not None:
        kwargs["bitrate"] = args.audio_bitrate

    app = ScreenRecorder(**kwargs)
    app.run()
