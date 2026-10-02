# Stay Fair!

Official implementation of [**"Stay Fair! Ensuring Group Fairness in Diffusion Models Across Guidance Scales"**](https://arxiv.org/abs/2605.28036).

**Authors:** Myeongsoo Kim, Eunji Kim, Minwoo Chae, Sangwoo Mo

Code will be released soon.

## Debiased models

`generate/debiased/` applies StayFair on top of SD1.5 debiased with UCE or FT.

- **UCE** ([Gandikota et al., 2024](https://github.com/rohitgandikota/unified-concept-editing)):
  train a gender-debiased SD1.5 checkpoint with the UCE repository and pass it with `--uce_model_path`.
- **FT** ([Shen et al., 2024](https://github.com/sail-sg/finetune-fair-diffusion)):
  download the `exp-1-debias-gender` checkpoint from that repository, unzip it to
  `exp-1-debias-gender/outputs/`, and pass
  `from-paper_finetune-text-encoder_09190215/checkpoint-9800_exported/text_encoder_lora_EMA.pth`
  with `--lora_path`.

## Citation

Coming soon.
