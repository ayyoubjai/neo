import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tool_runtime.desktop import DesktopError
from tool_runtime.wayland_input import WaylandInput, REMOTE, CAST
from tool_runtime.tools import computer_click, computer_screenshot, ToolError


class WaylandInputTests(unittest.TestCase):
    def setUp(self):
        self.connection = Mock()
        self.connection.GLib = SimpleNamespace(Variant=lambda signature, value: value)
        self.connection.monitor_size.return_value = (1280, 720)
        self.connection.request.side_effect = [
            {'session_handle': '/session/test'}, {}, {},
            {'devices': 3, 'streams': [(7, {'size': (1280, 720)})]},
        ]
        self.backend = WaylandInput(self.connection, capture_size=lambda: (1920, 1080))
        self.addCleanup(self.backend.close)

    def test_session_negotiates_before_input_and_is_reused(self):
        self.backend.click(960, 540)
        self.backend.click(600, 300)
        self.assertEqual([c.args[:2] for c in self.connection.request.call_args_list],
                         [(REMOTE, 'CreateSession'), (REMOTE, 'SelectDevices'),
                          (CAST, 'SelectSources'), (REMOTE, 'Start')])
        calls = self.connection.call.call_args_list
        self.assertEqual(calls[0].args[1], 'NotifyPointerMotionAbsolute')
        self.assertEqual(calls[0].args[3], ('/session/test', {}, 7, 640.0, 360.0))
        self.assertEqual(calls[1].args[3][-2:], (272, 1))
        self.assertEqual(calls[2].args[3][-2:], (272, 0))

    def test_denial_closes_session_and_sends_no_input(self):
        self.connection.request.side_effect = [
            {'session_handle': '/session/test'}, {}, {},
            DesktopError('Denied', 'PERMISSION_DENIED')]
        with self.assertRaises(DesktopError):
            self.backend.click(10, 10)
        self.connection.call.assert_not_called()
        self.connection.close_path.assert_called_once_with('/session/test')

    def test_mismatched_monitor_does_not_send_input(self):
        self.connection.monitor_size.return_value = (2560, 1440)
        with self.assertRaisesRegex(DesktopError, 'geometry'):
            self.backend.click(10, 10)
        self.connection.call.assert_not_called()

    def test_out_of_bounds_and_missing_capture_rejected(self):
        with self.assertRaisesRegex(DesktopError, 'outside'):
            self.backend.click(1920, 10)
        self.backend.capture_size = lambda: None
        with self.assertRaisesRegex(DesktopError, 'Capture a full screenshot'):
            self.backend.click(10, 10)
        self.connection.call.assert_not_called()

    def test_revocation_does_not_reopen_or_replay(self):
        self.connection.call.side_effect = DesktopError('Revoked', 'PERMISSION_DENIED')
        with self.assertRaisesRegex(DesktopError, 'Revoked'):
            self.backend.click(10, 10)
        with self.assertRaisesRegex(DesktopError, 'Revoked'):
            self.backend.click(10, 10)
        self.assertEqual(self.connection.request.call_count, 4)
        self.connection.call.assert_called_once()

    def test_drag_releases_button_when_motion_fails(self):
        self.backend.prepare()
        with patch.object(self.backend, 'moveTo', side_effect=ValueError('bad motion')):
            with self.assertRaises(ValueError):
                self.backend.dragTo(100, 100)
        self.assertEqual([c.args[3][-1] for c in self.connection.call.call_args_list], [1, 0])

    def test_hotkey_releases_in_reverse_order_and_scroll_direction(self):
        with patch.object(self.backend, '_symbol', side_effect=[10, 20]):
            self.backend.hotkey('ctrl', 'a')
        keys = [c.args[3][-2:] for c in self.connection.call.call_args_list]
        self.assertEqual(keys, [(10, 1), (20, 1), (20, 0), (10, 0)])
        self.backend.scroll(3)
        self.assertEqual(self.connection.call.call_args.args[3][-2:], (0, -3))

    def test_keyboard_permission_is_checked(self):
        self.backend.prepare()
        self.backend.devices = 2
        with self.assertRaisesRegex(DesktopError, 'Keyboard control'):
            self.backend._key(20, 1)
        self.connection.call.assert_not_called()

    def test_tool_error_preserves_portal_denial(self):
        with patch('tool_runtime.tools._load_computer_control_backend', return_value=self.backend):
            self.connection.request.side_effect = DesktopError('User declined', 'PERMISSION_DENIED')
            with self.assertRaises(ToolError) as error:
                computer_click({'x': 10, 'y': 10}, '.')
        self.assertEqual(error.exception.details['code'], 'PERMISSION_DENIED')

    def test_permission_is_requested_before_screenshot(self):
        with patch('tool_runtime.tools.is_wayland', return_value=True), \
                patch('tool_runtime.tools._load_computer_control_backend', return_value=self.backend), \
                patch('tool_runtime.tools.capture_screen') as capture:
            self.connection.request.side_effect = DesktopError('Denied', 'PERMISSION_DENIED')
            with self.assertRaises(ToolError):
                computer_screenshot({'prepare_input': True}, '.')
            capture.assert_not_called()
