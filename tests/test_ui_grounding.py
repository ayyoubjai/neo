from types import SimpleNamespace
from unittest.mock import patch

from model_server import uground_model


def _settings(**models):
    defaults = {
        'vision_model': 'qwen-vl.gguf',
        'ui_grounding_enforce_json': True,
        'ui_grounding_disable_thinking': True,
        'ui_grounding_use_thinking_on_empty': False,
        'ui_grounding_model_options': {},
        'llamacpp_timeout_s': 60,
    }
    defaults.update(models)
    return SimpleNamespace(models=defaults)


def test_llamacpp_grounding_uses_vision_fallback_and_chat_extraction():
    with patch.object(uground_model, 'load_settings', return_value=_settings()), \
         patch.object(uground_model, '_load_image_bytes', return_value=b'image'), \
         patch.object(uground_model, '_image_size', return_value=(1920, 1080)), \
         patch.object(uground_model, 'llamacpp_generate', return_value='{"x":750,"y":820,"confidence":0.9}') as generate, \
         patch.object(uground_model, 'record_llm_json'), \
         patch.object(uground_model, 'record_event'):
        result = uground_model._analyze_llamacpp('workspace:/screen.png', 'like button')
    assert result['ok'] is True
    assert result['json'] == {'x': 750.0, 'y': 820.0, 'confidence': 0.9}
    assert generate.call_args.args[1] == 'qwen-vl.gguf'
    assert generate.call_args.kwargs['response_format'] == 'json'
    assert 'entire supplied image' in generate.call_args.args[0]


def test_llamacpp_grounding_rejects_descriptive_json_without_point():
    with patch.object(uground_model, 'load_settings', return_value=_settings()), \
         patch.object(uground_model, '_load_image_bytes', return_value=b'image'), \
         patch.object(uground_model, '_image_size', return_value=(1920, 1080)), \
         patch.object(uground_model, 'llamacpp_generate', return_value='{"answer":"button visible"}'), \
         patch.object(uground_model, 'log_error'):
        result = uground_model._analyze_llamacpp('workspace:/screen.png', 'like button')
    assert result['ok'] is False
    assert 'numeric x and y' in result['error']


def test_point_validation_rejects_non_normalized_coordinates():
    payload, error = uground_model._validate_point({'x': 1500, 'y': 20})
    assert payload == {}
    assert '0-1000' in error
