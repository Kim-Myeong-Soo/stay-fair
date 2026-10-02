<div align="center">

# Stay Fair! Ensuring Group Fairness in Diffusion Models<br>Across Guidance Scales

[![arXiv](https://img.shields.io/badge/arXiv-2605.28036-b31b1b.svg)](https://arxiv.org/abs/2605.28036)
[![PDF](https://img.shields.io/badge/PDF-Download-FF6F00.svg)](https://arxiv.org/pdf/2605.28036)

Myeongsoo Kim, Eunji Kim, Minwoo Chae, [Sangwoo Mo](https://sites.google.com/view/sangwoomo)

**NeurIPS 2026**

</div>

> [!NOTE]
> This README is a work in progress.

## Overview

## Environments

Each model uses its own environment: `pip install -r envs/<sd15|sd3|flux2|debiased|eval>.txt`.

## Generation

| Script | Model | Paper setting |
|---|---|---|
| `generate/vanilla/gen_image_sd15.py` | SD1.5 | w 2.5–12.5, 30 steps, 512px |
| `generate/vanilla/gen_image_sd3.py` | SD3-medium | w 1.5–7.5, 28 steps, 512px |
| `generate/vanilla/gen_image_flux2.py` | FLUX.2 [klein] 9B | w 1.0–5.0, 28 steps, 768px |
| `generate/debiased/gen_image_sd15_uce_wref.py` | SD1.5 + UCE | w 2.5–12.5, w_ref 7.5 |
| `generate/debiased/gen_image_sd15_fair.py` | SD1.5 + FT | w 2.5–12.5, w_fair 7.5 |

`--test_option cfg` runs CFG and `--test_option cfg_gd --gd_json data/alpha/<model>.json` runs StayFair with the per-occupation α used in the paper. Prompts are in `data/occupation_8_v{1-4}.json` (33 occupations, 4 templates).

```bash
python generate/vanilla/gen_image_sd3.py --guidance_scale 4.5 --save_dir outputs/sd3/cfg/w4.5 --test_option cfg
python generate/vanilla/gen_image_sd3.py --guidance_scale 4.5 --save_dir outputs/sd3/stayfair/w4.5 \
    --test_option cfg_gd --gd_json data/alpha/sd3.json
```

StayFair (Estimate) for SD3: `estimate/build_lookup.py` builds the lookup table (`data/estimate/sd3_lookup_n5.csv`) and `estimate/apply_lookup.py` turns CFG probe counts at w = 1.5 and 7.5 into a `--gd_json`.

## Evaluation

```bash
python eval/eval_person.py --root outputs/sd3/stayfair   # YOLO person detection
python eval/eval_gender.py --root outputs/sd3/stayfair   # CLIP zero-shot gender
python eval/aggregate_gender.py --roots outputs/sd3/stayfair --out counts.csv
```

`eval_clip_score.py`, `eval_aesthetic.py`, and `eval_pickscore.py` measure alignment and quality. Female ratios per guidance scale for the paper's CFG and StayFair rows are in `results/female_ratio_<model>.csv`.

## Checkpoints

- **UCE**: train a gender-debiased SD1.5 with [UCE](https://github.com/rohitgandikota/unified-concept-editing) and pass it with `--uce_model_path`.
- **FT**: download the `exp-1-debias-gender` checkpoint from [Finetuning Fair Diffusion](https://github.com/sail-sg/finetune-fair-diffusion), unzip it to `exp-1-debias-gender/outputs/`, and pass `from-paper_finetune-text-encoder_09190215/checkpoint-9800_exported/text_encoder_lora_EMA.pth` with `--lora_path`.

## Citation

```bibtex
@misc{kim2026stayfairensuringgroup,
      title={Stay Fair! Ensuring Group Fairness in Diffusion Models Across Guidance Scales}, 
      author={Myeongsoo Kim and Eunji Kim and Minwoo Chae and Sangwoo Mo},
      year={2026},
      eprint={2605.28036},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.28036}, 
}
```
