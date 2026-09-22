"""Supervise a swapping GPU router and an optional dedicated CPU embedder."""
import argparse
import configparser
import json
from pathlib import Path
import signal
import subprocess
import time
from urllib.parse import urlsplit
import urllib.request
import urllib.error
from common.config import resolve_settings_path


def positive_int(value, name):
    result = int(value)
    if result < 1:
        raise ValueError(f'{name} must be a positive integer')
    return result


def server_commands(models, executable, preset, host, port):
    router = [executable, '--models-preset', str(preset), '--host', host, '--port', str(port),
              '--models-max', str(positive_int(models.get('router_max_models', 1), 'router_max_models')),
              '--models-autoload']
    commands = [('router', router)]
    embedding_host = models.get('embedding_host')
    if embedding_host:
        address = urlsplit(embedding_host)
        if address.scheme != 'http' or address.hostname not in ('127.0.0.1', 'localhost') or not address.port or address.path not in ('', '/'):
            raise ValueError('Managed embedding_host must be a local http://127.0.0.1:PORT address')
        if address.port == int(port):
            raise ValueError('CPU embeddings and GPU router must use different ports')
        ini = configparser.ConfigParser(interpolation=None)
        ini.read_string('[DEFAULT]\n' + Path(preset).read_text(encoding='utf-8'))
        name = models.get('embedding_model', '')
        model = ini.get(name, 'model')
        cpu = [executable, '--model', model, '--alias', name, '--embeddings',
               '--host', '127.0.0.1', '--port', str(address.port),
               '--device', 'none', '--n-gpu-layers', '0', '--parallel', '1',
               '--ctx-size', '2048', '--threads',
               str(positive_int(models.get('embedding_threads', 4), 'embedding_threads'))]
        # Start the CPU service before advertising router readiness.
        commands.insert(0, ('cpu-embeddings', cpu))
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-bin', required=True)
    parser.add_argument('--models-preset', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    models = json.loads(Path(resolve_settings_path()).read_text(encoding='utf-8')).get('models', {})
    commands = server_commands(models, args.server_bin, args.models_preset, args.host, args.port)
    processes = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for name, command in commands:
            print(f'[models] Starting {name}: {command}', flush=True)
            processes.append(subprocess.Popen(command))
            if name == 'cpu-embeddings':
                deadline = time.monotonic() + 60
                while True:
                    if processes[-1].poll() is not None:
                        raise RuntimeError('CPU embedding service exited during startup')
                    try:
                        with urllib.request.urlopen(models['embedding_host'].rstrip('/') + '/health', timeout=1) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError('CPU embedding service did not become ready in 60 seconds')
                    time.sleep(0.2)
        while all(process.poll() is None for process in processes):
            time.sleep(0.2)
        raise RuntimeError('A model service exited; stopping the managed model stack')
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
