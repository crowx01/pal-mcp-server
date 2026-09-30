from __main__ import * 
import os, json, logging
from enum import Enum
from providers.openai_compatible import OpenAICompatibleProvider
from providers.base import ProviderType, ModelCapabilities, RangeTemperatureConstraint

logger = logging.getLogger(__name__)

def _add_enum_member(enum_cls, name, value):
    upper_name = name.upper()
    if upper_name in enum_cls.__members__:
        return enum_cls[upper_name]
    try:
        import aenum
        return aenum.extend_enum(enum_cls, upper_name, value)
    except Exception:
        pass
    # Manual extension for standard Enum
    obj = object.__new__(enum_cls)
    obj._name_ = upper_name
    obj._value_ = value
    enum_cls._member_map_[upper_name] = obj
    enum_cls._value2member_map_[value] = obj
    enum_cls._member_names_.append(upper_name)
    setattr(enum_cls, upper_name, obj)
    return obj

def register_extras(registry):
    keys_path = os.path.expanduser("~/.pal/keys.json")
    if not os.path.exists(keys_path):
        return
    try:
        with open(keys_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.debug(f"Failed to load keys.json: {e}")
        return

    skipped_names = {"openrouter", "gemini", "groq", "openai", "custom", "huggingface", "cohere", "glm"}
    
    providers_dict = data if isinstance(data, dict) else {}
    for name, info in providers_dict.items():
        if not isinstance(info, dict):
            continue
        if name.lower() in skipped_names:
            continue
        status = str(info.get("status", ""))
        if not status.startswith("WORKING"):
            continue
        base = info.get("base")
        if not base:
            continue
        keys = info.get("keys", [])
        if not keys or not keys[0]:
            continue
        api_key = keys[0]

        try:
            ptype = _add_enum_member(ProviderType, name, name.lower())
            
            class DynamicOpenAIProvider(OpenAICompatibleProvider):
                BASE_URL = base
                FRIENDLY_NAME = name
                def get_provider_type(self):
                    return ptype
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
                    return True

            inst = DynamicOpenAIProvider(api_key=api_key)
            registry.register_provider(ptype, inst)
            
            if ptype in registry.PROVIDER_PRIORITY_ORDER:
                registry.PROVIDER_PRIORITY_ORDER.remove(ptype)
            if ProviderType.OPENROUTER in registry.PROVIDER_PRIORITY_ORDER:
                idx = registry.PROVIDER_PRIORITY_ORDER.index(ProviderType.OPENROUTER)
                registry.PROVIDER_PRIORITY_ORDER.insert(idx, ptype)
            else:
                registry.PROVIDER_PRIORITY_ORDER.insert(0, ptype)
        except Exception as e:
            logger.debug(f"Failed to register extra provider {name}: {e}")
