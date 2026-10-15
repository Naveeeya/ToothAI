# Photo screening: measured results

Generated 15 October 2026 by `scripts/write_eval_md.py` from measured files only. Every number below names the data split it comes from. Model versions (SHA-256 prefix): `alphadent` 361b05760577, `caries_aku` 553193c807ca, `teeth_dentalai` 5d2423634384.

## End to end on smartphone photos (AKU official test split)

The service as configured (`service default (+ cavity/crack off)`), run on 628 test photos (226 with caries, 396 without; 6 rejected by the quality gate). 95 % Wilson intervals in brackets.

| Measure | Value |
| --- | --- |
| Photos with caries that get a check-up or dentist referral | 88.1 % [83.2, 91.7] |
| Caries-free photos referred anyway (false referral) | 4.0 % [2.5, 6.5] |
| Image-level caries sensitivity | 89.4 % [84.7, 92.8] |
| Image-level caries specificity | 95.5 % [92.9, 97.1] |
| Caries AP@50 / box precision / box recall | 0.868 / 0.818 / 0.845 |
| Findings per photo (all) / Medium or High | 2.96 / 2.27 |

In plain terms: of 100 people whose photos show decay, about 88 are told to book a check-up or see a dentist; of 100 whose photos show none, about 4 are told to anyway. Decay between teeth or below the gum line can't be seen in photos at all, which is why the report recommends an OPG.

**Configurations compared on the validation split** (chosen there, then reported on test above):

| Configuration | Referral with caries | False referral | Specificity | Findings per photo |
| --- | --- | --- | --- | --- |
| original (AlphaDent caries standalone, cavity/crack on, wear from 0.30) | 93.7 % | 12.0 % | 79.4 % | 3.89 |
| AlphaDent caries as agreement only | 92.7 % | 7.9 % | 92.8 % | 3.73 |
| + wear from 0.55 | 92.7 % | 7.9 % | 92.8 % | 3.02 |
| service default (+ cavity/crack off) | 92.7 % | 6.5 % | 92.8 % | 2.91 |

## Per model

| Model | Measure | Value | Split |
| --- | --- | --- | --- |
| `caries_aku` | mAP@50, mean of `d`/`D` classes | 0.764 (published YOLOv8s 0.841) | AKU test, 628 images |
| `caries_aku` | AP@50, classes merged | 0.895 | same |
| `alphadent` | Caries AP@50 on phone photos (cross-domain) | 0.006 | same |
| `alphadent` | Caries image sensitivity / specificity at 0.25 | 64.6 % / 59.0 % | same |
| `teeth_dentalai` | tooth: AP@50; precision / recall at keep 0.17 | 0.739; 0.56 / 0.85 | DentalAI test, 1699 boxes |
| `teeth_dentalai` | cavity: AP@50; precision / recall at keep 0.25 | 0.255; 0.34 / 0.25 | DentalAI test, 93 boxes |
| `teeth_dentalai` | crack: AP@50; precision / recall at keep 0.3 | 0.357; 1.00 / 0.11 | DentalAI test, 9 boxes |
| `teeth_dentalai` | tooth mask AP@50 (notebook) | 0.775 | DentalAI test |

## Quality gate and speed

- Real AKU test photos rejected by the hard checks: 1.0 % (resolution 0, blur 6, exposure 0).
- Inference per photo (cuda:0 (Kaggle T4)): `alphadent` 0.13 s, `caries_aku` 0.02 s, `teeth_dentalai` 0.03 s.

## Limitations

- AKU labels caries only, so wear, fillings and crowns from AlphaDent are not validated on phone photos.
- DentalAI tooth instances come from semantic masks split by watershed; tooth numbers are approximate.
- Cavity and crack: too weak (or too few labels) to report; off by default.
- All photos come from two public datasets; real users' phones, lighting and ages will differ.
