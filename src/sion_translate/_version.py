"""Release identities for the Python package and newly trained model artifacts."""

__version__ = "1.5.0"

# The Python distribution follows semantic versioning, while model artifacts use
# the public major.minor generation.  Keep the model generation explicit: old
# weights retain the version embedded in their export metadata during conversion.
MODEL_RELEASE_VERSION = "1.5"


__all__ = ["MODEL_RELEASE_VERSION", "__version__"]
