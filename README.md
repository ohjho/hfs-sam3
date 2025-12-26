---
title: SAM3
emoji: 📚
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.32.0
app_file: app.py
pinned: false
short_description: SAM3 Video Inference on ZeroGPU
---

# SAM3 HuggingFace Space Demo
with inspiration from [prithivMLmods' demo](https://huggingface.co/spaces/prithivMLmods/SAM3-Demo), using the [transformers API](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video)

# Requirements
using `git+https://github.com/huggingface/transformers.git` for now since it's not yet available on the latest release of transformers (v4.57.3 at the time of writing)

# The HuggingFace Space Template
setup with [github action to update automatically update your space](https://huggingface.co/docs/hub/spaces-github-actions)
and manage dependencies with `uv`

You will need to update [`deploy_to_hf_space.yaml`](.github/workflows/deploy_to_hf_space.yaml) with the details for your space and
setup your `HF_TOKEN` and `FORCE_PUSH` in your [Github secret](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-an-environment)

> [!WARNING]
> The Githuh Action *Force* push changes to HuggingFace Space
> This is due to the creation of the requirements.txt that happens on the fly.
> This template assumes that you are the sole contributor to your space.

## Resources

* [Gradio Course](https://huggingface.co/learn/llm-course/chapter9/2?fw=pt)
* [Gradio Doc](https://www.gradio.app/guides/quickstart)
* Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
