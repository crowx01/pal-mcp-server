from __main__ import * 
import os, json
from enum import Enum
try:
    import aenum
except ImportError:
    aenum = None

from providers.base import ProviderType, ModelCapabilities, RangeTemperatureConstraint
from providers.openai_compatible import OpenAICompatibleProvider

def register_extras(registry):
    keys_path = os.path.expanduser('~/.pal/keys.json')
    if not os.path.exists(keys_path):
        return
    try:
        with open(keys_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return

    skip_names = {'openrouter', 'gemini', 'groq', 'openai', 'custom', 'huggingface', 'cohere', 'glm'}

    for name, info in data.items():
        if name.lower() in skip_names:
            continue
        if not isinstance(info, dict):
            continue
        base = info.get('base')
        status = info.get('status', '')
        if not base or not status.startswith('WORKING'):
            continue
        keys = info.get('keys', [])
        if not keys:
            continue
        api_key = keys[0]

        upper_name = name.upper()
        lower_name = name.lower()

        # 3. Extend ProviderType Enum dynamically
        if not hasattr(ProviderType, upper_name):
            if aenum and hasattr(aenum, 'extend_enum'):
                try:
                    aenum.extend_enum(ProviderType, upper_name, lower_name)
                except Exception:
                    pass
            if not hasattr(ProviderType, upper_name):
                try:
                    ProviderType._member_map_[upper_name] = lower_name
                    ProviderType._value2member_map_[lower_name] = getattr(ProviderType, upper_name, None) or object()
                    # Actually assign as attribute to the Enum class if possible
                    setattr(ProviderType, upper_name, lower_name)
                except Exception:
                    pass

        p_type = getattr(ProviderType, upper_name, lower_name)

        # Create dynamic provider class
        class DynamicOpenAIProvider(OpenAICompatibleProvider):
            FRIENDLY_NAME = name
            def __init__(self, key=None):
                super().__init__(api_key=api_key)
                self.base_url = base

            @classmethod
            def get_provider_type(cls):
                return p_type

            def _lookup_capabilities(self, model_name: str) -> ModelCapabilities:
                return ModelCapabilities(
                    intelligence_score=9,
                    context_window=128000,
                    max_output_tokens=4096,
                    supports_system_prompts=True,
                    supports_streaming=True,
                    supports_function_calling=True,
                    temperature_constraint=RangeTemperatureConstraint(0, 2, 1)
                )

            def validate_model(self, model_name: str) -> bool:
                return bool(model_name)

        instance = DynamicOpenAIProvider()
        # Register provider class and instance
        registry.register_provider(p_type, DynamicOpenAIProvider)
        if hasattr(registry, 'register_instance'):
            registry.register_instance(p_type, instance)
        if hasattr(registry, '_instances'):
            registry._instances[p_type] = instance

        # Insert into PROVIDER_PRIORITY_ORDER before OPENROUTER
        if hasattr(registry, 'PROVIDER_PRIORITY_ORDER'):
            if p_type in registry.PROVIDER_PRIORITY_ORDER:
                registry.PROVIDER_PRIORITY_ORDER.remove(p_type)
            if ProviderType.OPENROUTER in registry.PROVIDER_PRIORITY_ORDER:
                idx = registry.PROVIDER_PRIORITY_ORDER.index(ProviderType.OPENROUTER)
                registry.PROVIDER_PRIORITY_ORDER.insert(idx, p_type)
            else:
                registry.PROVIDER_PRIORITY_ORDER.insert(0, p_type)
