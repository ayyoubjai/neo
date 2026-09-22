"""Consent-based Wayland input through XDG RemoteDesktop and ScreenCast.

Uses a dedicated D-Bus connection and GLib context so synchronous tool calls
do not depend on a GUI application's event loop. No input is sent before Start
grants the requested devices. The desktop owns and can revoke the session.
"""
import atexit
import math
import os
import sys
import threading
import time
import uuid

from tool_runtime.desktop import DesktopError


DEST = 'org.freedesktop.portal.Desktop'
PATH = '/org/freedesktop/portal/desktop'
REMOTE = 'org.freedesktop.portal.RemoteDesktop'
CAST = 'org.freedesktop.portal.ScreenCast'
REQUEST = 'org.freedesktop.portal.Request'
SESSION = 'org.freedesktop.portal.Session'


def _number(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise DesktopError(f'{name} must be finite.', 'MISSING_INPUT')
    return result


class PortalConnection:
    def __init__(self):
        try:
            import gi
            gi.require_version('Gdk', '3.0')
            from gi.repository import Gio, GLib, Gdk
        except (ImportError, ValueError) as exc:
            raise DesktopError(
                f'Wayland input dependencies are unavailable in {sys.executable}: {exc}. '
                'Install requirements-wayland.txt using this same Python interpreter. '
                'GTK 3 introspection and your desktop portal backend must also be installed on the system.',
                'MISSING_DEPENDENCY') from exc
        self.Gio, self.GLib, self.Gdk = Gio, GLib, Gdk
        self.context = GLib.MainContext.new()
        try:
            address = Gio.dbus_address_get_for_bus_sync(Gio.BusType.SESSION, None)
            self.bus = Gio.DBusConnection.new_for_address_sync(address,
                Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,
                None, None)
        except GLib.Error as exc:
            raise DesktopError(f'Cannot connect to the desktop session bus: {exc}', 'UNAVAILABLE_SOURCE') from exc
        self.bus.set_exit_on_close(False)

    def call(self, interface, method, signature, args, path=PATH):
        try:
            return self.bus.call_sync(DEST, path, interface, method,
                self.GLib.Variant(signature, args), None, self.Gio.DBusCallFlags.NONE,
                10000, None).unpack()
        except self.GLib.Error as exc:
            name = self.Gio.DBusError.get_remote_error(exc) or ''
            code = ('PERMISSION_DENIED' if 'AccessDenied' in name or 'NotAllowed' in name
                    else 'UNSUPPORTED_PLATFORM' if 'UnknownMethod' in name or 'UnknownInterface' in name
                    else 'UNAVAILABLE_SOURCE')
            raise DesktopError(f'Wayland portal {method} failed: {exc}', code) from exc

    def request(self, interface, method, prefix_signature='', prefix=(), options=None, timeout=60):
        token = 'neo_' + uuid.uuid4().hex
        options = dict(options or {})
        options['handle_token'] = self.GLib.Variant('s', token)
        sender = self.bus.get_unique_name()[1:].replace('.', '_')
        path = PATH + '/request/' + sender + '/' + token
        response = []
        def received(connection, sender, object_path, interface, signal, parameters, user_data):
            response.append(parameters.unpack())
        self.context.push_thread_default()
        subscription = self.bus.signal_subscribe(DEST, REQUEST, 'Response', path, None,
            self.Gio.DBusSignalFlags.NONE, received, None)
        try:
            returned_path, = self.call(interface, method, '(' + prefix_signature + 'a{sv})',
                                       (*prefix, options))
            if returned_path != path:
                self.bus.signal_unsubscribe(subscription)
                path = returned_path
                subscription = self.bus.signal_subscribe(DEST, REQUEST, 'Response', path, None,
                    self.Gio.DBusSignalFlags.NONE, received, None)
            deadline = time.monotonic() + timeout
            while not response:
                self.pump()
                if time.monotonic() >= deadline:
                    self.close_path(path, REQUEST)
                    raise DesktopError('Desktop permission dialog timed out. Retry and approve the portal prompt.',
                                       'PERMISSION_DENIED')
                if not response:
                    time.sleep(.01)
            status, result = response[0]
            if status:
                raise DesktopError('Desktop control was cancelled or denied in the portal dialog.', 'PERMISSION_DENIED')
            return result
        finally:
            self.bus.signal_unsubscribe(subscription)
            self.context.pop_thread_default()

    def pump(self):
        while self.context.pending():
            self.context.iteration(False)

    def monitor_size(self):
        # Force Wayland: XWayland's logical monitor layout may be different.
        self.Gdk.set_allowed_backends('wayland')
        display = self.Gdk.Display.open(os.environ.get('WAYLAND_DISPLAY') or 'wayland-0')
        if display is None:
            raise DesktopError('Cannot inspect the Wayland monitor layout.', 'UNAVAILABLE_SOURCE')
        try:
            if display.get_n_monitors() != 1:
                raise DesktopError('Wayland absolute input currently supports one monitor. '
                                   'Use a single-monitor layout for screenshot-based clicks.', 'UNSUPPORTED_PLATFORM')
            geometry = display.get_monitor(0).get_geometry()
            return (geometry.width, geometry.height)
        finally:
            display.close()

    def close_path(self, path, interface=SESSION):
        try:
            self.call(interface, 'Close', '()', (), path=path)
        except DesktopError:
            pass

    def close(self):
        self.bus.close_sync(None)


class WaylandInput:
    """Small PyAutoGUI-compatible surface used by Neo's computer tools."""
    def __init__(self, connection=None, capture_size=None):
        self.connection = connection or PortalConnection()
        self.capture_size = capture_size or (lambda: None)
        self.session = None
        self.stream = None
        self.devices = 0
        self.size = None
        self.position = None
        self.failure = None
        self._lock = threading.RLock()
        atexit.register(self.close)

    def prepare(self):
        with self._lock:
            if self.failure:
                raise self.failure
            if self.session:
                return
            c = self.connection
            monitor_size = c.monitor_size()
            v = c.GLib.Variant
            result = c.request(REMOTE, 'CreateSession', options={'session_handle_token': v('s', 'neo_' + uuid.uuid4().hex)})
            session = result['session_handle']
            try:
                c.request(REMOTE, 'SelectDevices', 'o', (session,), {'types': v('u', 3)})
                c.request(CAST, 'SelectSources', 'o', (session,), {'types': v('u', 1), 'multiple': v('b', False)})
                result = c.request(REMOTE, 'Start', 'os', (session, ''))
                streams = result.get('streams', [])
                if len(streams) != 1 or not (int(result.get('devices', 0)) & 2):
                    raise DesktopError('Select your monitor and allow pointer control in the portal dialog.', 'PERMISSION_DENIED')
                node, props = streams[0]
                size = tuple(props.get('logical_size') or props.get('size') or ())
                if size != monitor_size:
                    raise DesktopError('Portal monitor geometry does not match the desktop. '
                                       'Select the physical monitor, not a virtual display.', 'UNSUPPORTED_PLATFORM')
                self.session, self.stream, self.size = session, int(node), size
                self.devices = int(result['devices'])
            except Exception:
                c.close_path(session)
                raise

    def _send(self, method, suffix, *values):
        self.prepare()
        try:
            self.connection.call(REMOTE, method, '(oa{sv}' + suffix + ')', (self.session, {}, *values))
        except DesktopError as exc:
            # Never silently reopen a revoked session or replay an input event.
            self.failure = exc
            self.close_session()
            raise

    def close_session(self):
        if self.session:
            self.connection.close_path(self.session)
        self.session = self.stream = self.size = self.position = None

    def close(self):
        with self._lock:
            self.close_session()
            try:
                self.connection.close()
            except Exception:
                pass
        atexit.unregister(self.close)

    def _point(self, x, y):
        self.prepare()
        if self.connection.monitor_size() != self.size:
            self.close_session()
            raise DesktopError('Monitor layout changed. Capture a new screenshot before retrying.', 'MISSING_INPUT')
        pixels = self.capture_size()
        if not pixels:
            raise DesktopError('Capture a full screenshot before using Wayland pointer coordinates.', 'MISSING_INPUT')
        x, y = _number(x, 'x'), _number(y, 'y')
        if not (0 <= x < pixels[0] and 0 <= y < pixels[1]):
            raise DesktopError('Pointer coordinates are outside the captured screen.', 'MISSING_INPUT')
        return x * self.size[0] / pixels[0], y * self.size[1] / pixels[1]

    def moveTo(self, x, y, duration=0):
        with self._lock:
            point = self._point(x, y)
            duration = max(0, min(_number(duration, 'duration'), 30))
            origin = self.position or point
            steps = max(1, int(duration * 30))
            for index in range(1, steps + 1):
                fraction = index / steps
                self._send('NotifyPointerMotionAbsolute', 'udd', self.stream,
                           origin[0] + (point[0] - origin[0]) * fraction,
                           origin[1] + (point[1] - origin[1]) * fraction)
                if duration:
                    time.sleep(duration / steps)
            self.position = point

    def _button(self, button, state):
        codes = {'left': 272, 'right': 273, 'middle': 274}
        if button not in codes:
            raise DesktopError('button must be left, right, or middle.', 'MISSING_INPUT')
        self._send('NotifyPointerButton', 'iu', codes[button], state)

    def click(self, x, y, clicks=1, interval=0, button='left'):
        with self._lock:
            interval = max(0, min(_number(interval, 'interval'), 10))
            if button not in ('left', 'right', 'middle') or not 1 <= clicks <= 100:
                raise DesktopError('Invalid mouse button or click count (1-100).', 'MISSING_INPUT')
            self.moveTo(x, y)
            for index in range(clicks):
                self._button(button, 1)
                try:
                    time.sleep(.02)
                finally:
                    self._button(button, 0)
                if index + 1 < clicks:
                    time.sleep(interval)

    def dragTo(self, x, y, duration=0, button='left'):
        with self._lock:
            self._point(x, y)  # Validate before pressing.
            self._button(button, 1)
            try:
                self.moveTo(x, y, duration)
            finally:
                self._button(button, 0)

    def scroll(self, amount):
        with self._lock:
            self._send('NotifyPointerAxisDiscrete', 'ui', 0, -int(amount))

    def _key(self, symbol, state):
        self.prepare()
        if not self.devices & 1:
            raise DesktopError('Keyboard control was not granted by the desktop portal.', 'PERMISSION_DENIED')
        self._send('NotifyKeyboardKeysym', 'iu', symbol, state)

    def _symbol(self, key):
        aliases = {'ctrl': 'Control_L', 'control': 'Control_L', 'shift': 'Shift_L', 'alt': 'Alt_L',
                   'win': 'Super_L', 'command': 'Super_L', 'super': 'Super_L', 'enter': 'Return',
                   'return': 'Return', 'esc': 'Escape', 'escape': 'Escape', 'tab': 'Tab',
                   'backspace': 'BackSpace', 'delete': 'Delete', 'space': 'space',
                   'left': 'Left', 'right': 'Right', 'up': 'Up', 'down': 'Down',
                   'home': 'Home', 'end': 'End', 'pageup': 'Page_Up', 'pagedown': 'Page_Down'}
        if len(key) == 1:
            return self.connection.Gdk.unicode_to_keyval(ord(key))
        name = aliases.get(key.lower(), key.upper() if key.lower().startswith('f') else key)
        symbol = self.connection.Gdk.keyval_from_name(name)
        if not symbol or symbol == 0xffffff:
            raise DesktopError(f'Unsupported keyboard key: {key}', 'MISSING_INPUT')
        return symbol

    def hotkey(self, *keys):
        with self._lock:
            symbols = [self._symbol(key) for key in keys]
            pressed = []
            try:
                for symbol in symbols:
                    self._key(symbol, 1)
                    pressed.append(symbol)
            finally:
                for symbol in reversed(pressed):
                    self._key(symbol, 0)

    def write(self, text, interval=0):
        with self._lock:
            interval = max(0, min(_number(interval, 'interval'), 10))
            for char in text:
                symbol = self._symbol({'\n': 'enter', '\r': 'enter', '\t': 'tab'}.get(char, char))
                self._key(symbol, 1)
                try:
                    if interval:
                        time.sleep(interval)
                finally:
                    self._key(symbol, 0)
