"""aidetect: a local-first, probabilistic AI-generated-text analysis system.

The system estimates whether text exhibits statistical patterns associated with
AI-generated writing. Its outputs are probabilistic indicators, never proof of
authorship.
"""

__version__ = "1.0.0"

# Bumped whenever the meaning or computation of any feature changes. Model bundles
# record the version they were trained with and refuse to run with another one.
FEATURE_VERSION = "2026.10-f3"

DISCLAIMER = (
    "AI-text detection is probabilistic. The results should not be used as sole "
    "evidence for academic misconduct, disciplinary action, or authorship "
    "determination. The detector is designed to assist human review, not to replace it."
)

INTERPRETATION = (
    "Based on the characteristics measured by this model, the text contains patterns "
    "that are {level} associated with AI-generated writing. This is a statistical "
    "estimate, not a determination of authorship."
)

SIGNAL_CAVEAT = "These are statistical indicators, not proof of AI authorship."

PRIVACY_NOTE = "Privacy mode: Local processing. No text leaves this computer."
