# Desktop tools on Linux and Windows

Run Neo from the logged-in user's interactive desktop and install
`requirements.txt` in the same Python environment used to launch it.

| Feature | Windows | Linux X11 | Linux Wayland |
| --- | --- | --- | --- |
| Screenshot / region capture | Pillow | Pillow | Spectacle, gnome-screenshot, or grim |
| Click, move, drag, type, hotkey, scroll | PyAutoGUI | PyAutoGUI | RemoteDesktop portal; one monitor |
| Focus window by title | PyGetWindow | xdotool or wmctrl | Requires an X11 login session |
| Launch executable with arguments | Supported | Supported | Supported |

Screenshot capture does not require PyAutoGUI. On Wayland, install the capture
utility appropriate for your compositor: Spectacle for KDE, gnome-screenshot
for GNOME, or grim for compatible wlroots compositors. Availability and capture
permission depend on the compositor. Neo tries installed alternatives after
failures, bounds each attempt to 20 seconds, and removes temporary capture files.
It does not fall back to an XWayland root-window capture, which can miss native
windows. A permission dialog may still be required by the desktop.

On Wayland, Neo requests mouse and keyboard control through the XDG RemoteDesktop
portal. Approve KDE/GNOME's dialog and select your monitor. The session is reused
while Neo runs and can be revoked through the desktop. Window focus by title is
still unsupported on Wayland. Absolute pointer coordinates currently require a
single monitor; multi-monitor layouts are rejected. Windows elevated applications
may reject input from a process with lower privileges.

Install the optional Python dependency into the **same environment that starts
Neo**. `start.sh` activates `.venv`, so system Python packages alone are insufficient:

```bash
.venv/bin/python -m pip install -r requirements-wayland.txt
.venv/bin/python -c "import gi; gi.require_version('Gdk', '3.0'); from gi.repository import Gio, GLib, Gdk; print('Wayland dependencies ready')"
```

GTK 3 introspection, GLib/GObject development libraries, Cairo and a RemoteDesktop
portal backend must exist on the system. They were already present on the tested
Arch/KDE machine. No root input daemon or unrestricted input-device access is used.

`computer.screenshot` returns an `image_ref`. Supply that actual value to
`image.analyse` in a subsequent action cycle; `<screenshot_path>` is not a result
reference. Describing the image additionally requires a configured vision model.
The `vision.observe` tool observes the camera stream, not the desktop.

For visual interaction, use `computer.screenshot`, then pass its `image_ref` and
a precise target description to `ui.predict_coords`, then pass the returned pixel
`x` and `y` to `computer.click`. `image.analyse` is for descriptions and OCR; its
output schema deliberately does not promise click coordinates. Tool prerequisites
make the grounding and screenshot tools available whenever a click needs visual
location, even if embedding retrieval omitted them initially.
Set `prepare_input: true` on that screenshot call to complete portal approval
before capturing the image used for coordinates. Ordinary descriptive screenshots
do not request input control. After permission revocation or a portal connection
failure, restart Neo to establish a fresh session; failed input is not replayed.

For llama.cpp, set `models.vision_model` and `models.vision_mmproj` in
`config/settings.local.json`. The projector path may be absolute or relative
to `models.models_dir`. The preset generator writes `mmproj` for the vision
model and its aliases. Regeneration also preserves existing per-model options
such as `mmproj` and `ctx-size`; an explicit `vision_mmproj` setting takes
precedence. This keeps vision configuration intact when `start.sh` regenerates
`config/models.ini`, and restores the projector even if that file is recreated.
`models.ui_grounding_model` can select a dedicated grounding model; if it is empty,
the llama.cpp adapter uses `models.vision_model`. Qwen-VL grounding presets default
to at least 1024 image tokens, configurable through `vision_image_min_tokens`.

Coordinates are pixels in the captured desktop. Mixed monitor scaling can
affect the relationship between screenshot coordinates and input coordinates;
verify positioning before running automated clicks. PyAutoGUI typing retains
its keyboard-layout and character-set limitations.
