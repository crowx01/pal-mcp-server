from __future__ import annotations

import json
import os
import logging
from pathlib import Path

from providers.base import ProviderType, ModelCapabilities, RangeTemperatureConstraint
from providers.openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

SKIP_NAMES = {"openrouter", "gemini", "groq", "openai", "custom", "huggingface", "cohere", "glm"}

def _extend_provider_type(name: str) -> ProviderType:
    upper_name = name.upper()
    lower_name = name.lower()
    if upper_name in ProviderType.__members__:
        return ProviderType[upper_name]
    
    try:
        import aenum
        aenum.extend_enum(ProviderType, upper_name, lower_name)
        return ProviderType[upper_name]
    except Exception:
        pass

    # Manual extension if aenum not available
    try:
        new_member = object.__new__(ProviderType)
        new_member._name_ = upper_name
        new_member._value_ = lower_name
        ProviderType._member_map_[upper_name] = new_member
        ProviderType._value2member_map_[lower_name] = new_member
        if new_member not in ProviderType._member_names_:
            ProviderType._member_names_.append(upper_name)
        return new_member
    except Exception as e:
        logger.debug(f"Failed to extend ProviderType for {name}: {e}")
        return ProviderType.CUSTOM

class GenericExtraProvider(OpenAICompatibleProvider):
    def __init__(self, ptype: ProviderType, friendly_name: str, base_url: str, api_key: str):
        super().__init__()
        self._ptype = ptype
        self._friendly_name = friendly_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def get_provider_type(self) -> ProviderType:
        return self._ptype

    @property
    def FRIENDLY_NAME(self) -> str:
        return self._friendly_name

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    def _lookup_capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(
            intelligence_score=9,
            context_window=128000,
            max_output_tokens=4096,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
        )

    def validate_model(self, model: str) -> bool:
        return bool(model and model.strip())

def register_extras(registry):
    keys_path = Path.home() / ".pal" / "keys.json"
    if not keys_path.exists():
        return
    try:
        data = json.loads(keys_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"Failed to read keys.json: {e}")
        return

    providers_dict = data.get("providers", data)
    if not isinstance(providers_dict, dict):
        return

    for name, info in providers_dict.items():
        if not isinstance(info, dict):
            continue
        if name.lower() in SKIP_NAMES:
            continue
        status = str(info.get("status", ""))
        if not status.startswith("WORKING"):
            continue
        base = info.get("base") or info.get("base_url")
        if not base:
            continue
        keys_list = info.get("keys", [])
        if not keys_list or not isinstance(keys_list, list):
            continue
        api_key = keys_list[0]
        if not api_key:
            continue

        try:
            ptype = _extend_provider_type(name)
            provider_inst = GenericExtraProvider(ptype, name, base, api_key)
            
            # Register instance directly if registry supports it, or store in registry cache
            if hasattr(registry, "register_provider_instance"):
                registry.register_provider_instance(ptype, provider_inst)
            elif hasattr(registry, "register_provider"):
                registry.register_provider(ptype, type(provider_inst))
                # Also populate instantiated cache if present
                if hasattr(registry, "_instances"):
                    registry._instances[ptype] = provider_inst

            if hasattr(registry, "PROVIDER_PRIORITY_ORDER"):
                if ptype in registry.PROVIDER_PRIORITY_ORDER:
                    registry.PROVIDER_PRIORITY_ORDER.remove(ptype)
                if ProviderType.OPENROUTER in registry.PROVIDER_PRIORITY_ORDER:
                    idx = registry.PROVIDER_PRIORITY_ORDER.index(ProviderType.OPENROUTER)
                    registry.PROVIDER_PRIORITY_ORDER.insert(idx, ptype)
                else:
                    registry.PROVIDER_PRIORITY_ORDER.insert(0, ptype)
            logger.info(f"Registered extra provider: {name} with base {base}")
        except Exception as e:
            logger.debug(f"Failed to register extra provider {name}: {e}")
