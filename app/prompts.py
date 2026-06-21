# WHY PROMPTS ARE EXTERNALIZED:
# Keeping prompts in separate .txt files (rather than Python string constants)
# makes prompt engineering changes visible in git diffs as plain text — easy
# to review, iterate on, and revert without touching application code.
# It also allows non-engineer stakeholders to review and edit prompt wording
# directly in the file without needing to understand Python.

import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


@lru_cache
def load_prompt(filename: str) -> str:
    """
    Load a prompt template from the app/prompts/ directory.
    Cached after first read so disk I/O only happens once per process.

    Args:
        filename: The name of the prompt file (e.g. "healthcare_rag_system_prompt.txt").

    Returns:
        The full text content of the prompt file.

    Raises:
        FileNotFoundError: If the prompt file does not exist in PROMPTS_DIR.
    """
    path = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(path):
        logger.error("Prompt file not found: %s", path)
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info("Loaded prompt template: %s (%d chars)", filename, len(content))
    return content
