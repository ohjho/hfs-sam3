---
title: Name for you Space App
emoji: 📚
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.32.0
app_file: app.py
pinned: false
short_description: short description for your Space App
---

# The HuggingFace Space Template
setup with [github action to update automatically update your space](https://huggingface.co/docs/hub/spaces-github-actions)
and manage dependencies with `uv`

You will need to update [`deploy_to_hf_space.yaml`](.github/workflows/deploy_to_hf_space.yaml) with the details for your space and
setup your `HF_TOKEN` in your [Github secret](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-an-environment)

> [!WARNING]
> The Githuh Action *Force* push changes to HuggingFace Space
> This is due to the creation of the requirements.txt that happens on the fly.
> This template assumes that you are the sole contributor to your space.

## Resources

* [Gradio Course](https://huggingface.co/learn/llm-course/chapter9/2?fw=pt)
* [Gradio Doc](https://www.gradio.app/guides/quickstart)
* Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
