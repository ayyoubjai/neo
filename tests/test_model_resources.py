from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading

import pytest

from model_server.managed_router import server_commands
from model_server.resource_lock import router_request_lock
from model_server import llamacpp_client
from model_server import managed_router


def test_router_and_cpu_embedder_have_separate_memory_budgets(tmp_path):
    preset = tmp_path / 'models.ini'
    preset.write_text('version = 1\n[embed]\nmodel = /models/embedding.gguf\n')
    commands = server_commands({'embedding_host': 'http://127.0.0.1:8082', 'embedding_model': 'embed'}, 'llama-server', preset, '127.0.0.1', 8080)
    cpu, gpu = commands[0][1], commands[1][1]
    assert cpu[cpu.index('--device') + 1] == 'none'
    assert cpu[cpu.index('--n-gpu-layers') + 1] == '0'
    assert '--embeddings' in cpu
    assert gpu[gpu.index('--models-max') + 1] == '1'
    assert '--models-dir' not in gpu  # No alternate presets bypassing policy.
    assert '--device' not in gpu  # Preserve automatic device selection.


def test_rejects_shared_cpu_gpu_port(tmp_path):
    with pytest.raises(ValueError, match='different ports'):
        server_commands({'embedding_host': 'http://127.0.0.1:8080'}, 'server', tmp_path, '127.0.0.1', 8080)


def test_requests_use_correct_server_and_lock(tmp_path):
    settings = SimpleNamespace(data_dir=str(tmp_path), models={
        'llamacpp_host': 'http://127.0.0.1:8080', 'embedding_host': 'http://127.0.0.1:8082',
        'serialize_model_requests': True,
    })
    with patch.object(llamacpp_client, 'load_settings', return_value=settings), patch.object(llamacpp_client, '_post_to_host', return_value={}) as post, patch.object(llamacpp_client, 'router_request_lock') as lock:
        llamacpp_client._post('/v1/embeddings', {})
        assert post.call_args.args[0] == settings.models['embedding_host']
        lock.assert_not_called()
        llamacpp_client._post('/v1/chat/completions', {})
        assert post.call_args.args[0] == settings.models['llamacpp_host']
        lock.assert_called_once()
        lock.return_value.__enter__.assert_called_once()


def test_request_lock_releases_after_exception_and_blocks_competitors(tmp_path):
    path = str(tmp_path / 'router.lock')
    results = []

    def compete():
        try:
            with router_request_lock(path, timeout=0.05):
                results.append('acquired')
        except TimeoutError:
            results.append('blocked')

    with pytest.raises(ValueError):
        with router_request_lock(path):
            thread = threading.Thread(target=compete)
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive()
            raise ValueError('Request failed')
    assert results == ['blocked']
    with router_request_lock(path, timeout=0.1):
        pass


def test_supervisor_stops_embedder_if_router_exits(tmp_path):
    settings = tmp_path / 'settings.json'
    settings.write_text('{"models":{"embedding_host":"http://127.0.0.1:18082"}}')
    cpu = Mock()
    cpu.poll.return_value = None
    router = Mock()
    router.poll.return_value = 1
    response = Mock(status=200)
    with patch.object(managed_router, 'resolve_settings_path', return_value=str(settings)), \
            patch.object(managed_router, 'server_commands', return_value=[('cpu-embeddings', ['cpu']), ('router', ['gpu'])]), \
            patch.object(managed_router.subprocess, 'Popen', side_effect=[cpu, router]), \
            patch.object(managed_router.signal, 'signal'), \
            patch.object(managed_router.urllib.request, 'urlopen') as urlopen, \
            patch('sys.argv', ['managed_router', '--server-bin', 'server', '--models-preset', 'preset']):
        urlopen.return_value.__enter__.return_value = response
        with pytest.raises(RuntimeError, match='model service exited'):
            managed_router.main()
    cpu.terminate.assert_called_once()
    cpu.wait.assert_called_once()
    router.terminate.assert_not_called()
