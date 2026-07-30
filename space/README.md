---
title: sion_translate Translator
emoji: 🇰🇷
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
models:
- gaon12/sion_translate
---

# sion_translate Translator

Interactive Korean↔Japanese translation with an immutable, SHA-256-verified
sion_translate EMA state-dictionary export loaded through PyTorch's weights-only mode.

The first request can take longer while the model is downloaded and loaded. Machine translation
can be wrong; review important output before use.

This directory is a deployment-ready Space scaffold. Hosting a Gradio or Docker Space on the
current Hugging Face `cpu-basic` plan requires a PRO subscription; it is not deployed by default.
