# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro.convsettings import (
    ConversationSettings,
    DEFAULT_MESSAGE_TEXT_MIN_SCORE,
    DEFAULT_RELATED_TERM_MIN_SCORE,
)


def test_conversation_settings_use_stricter_message_text_cutoff() -> None:
    settings = ConversationSettings(model=create_test_embedding_model())

    assert settings.related_term_index_settings.embedding_index_settings.min_score == (
        DEFAULT_RELATED_TERM_MIN_SCORE
    )
    assert settings.thread_settings.min_score == DEFAULT_RELATED_TERM_MIN_SCORE
    assert settings.message_text_index_settings.embedding_index_settings.min_score == (
        DEFAULT_MESSAGE_TEXT_MIN_SCORE
    )
