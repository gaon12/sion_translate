---
title: KJ-X Translator
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

# KJ-X Translator

Interactive Korean↔Japanese translation with the KJ-X INT8 CPU export.

The first request can take longer while the model is downloaded and loaded. Machine translation
can be wrong; review important output before use.

This directory is a deployment-ready Space scaffold. Hosting a Gradio or Docker Space on the
current Hugging Face `cpu-basic` plan requires a PRO subscription; it is not deployed by default.
