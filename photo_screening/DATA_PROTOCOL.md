# Data collection protocol: home smartphone photos (draft)

**Status: draft. No photo may be collected until the institutional ethics committee of Bharati Vidyapeeth
(Deemed to be University) Dental College & Hospital, Pune has approved this protocol in writing.** Record
the approval reference and date at the top of this file before collection starts. Any change to the
protocol after approval goes back to the committee first.

## Why

The caries model was trained on flash-lit clinical photos (AKU). On a real home set (warm indoor light,
fingers holding the lips) it scored obvious decay at 0.24 and missed it, and the restoration model read
fingernails as crowns. We have no labelled photos of that kind, so these failures can't be measured or
trained away. This protocol collects them.

## What we collect

- 5 intraoral photos per participant (front, upper, lower, left, right), taken by the participant with
  their own phone, at home, following only the in-app photo tips. Fingers in frame are expected.
- For each participant, a clinical examination by a dentist at the dental college within 14 days of the
  photos, recording decay, cavities, restorations and crowns per tooth (FDI). This is the reference.
- Age band (18–25, 26–40, 41–60, 60+) and phone model. Nothing else.

Not collected: names, faces beyond the mouth, dates of birth, contact details in the dataset, radiographs.

## Who

Adults (18+) who can consent for themselves, recruited at the dental college's outpatient department.
Target 150 participants (750 photos), roughly half with at least one carious lesion on examination.
Exclusions: anyone who declines the examination, and anyone under 18.

## Consent

Written informed consent, in English, Hindi or Marathi, covering: the purpose, that photos are used to
train and test a research screening model (not for diagnosis), who can see them, how long they are kept,
and that the participant can withdraw at any time before the data are frozen for a training run. Consent
forms are stored separately from the photos.

## Handling

- Each participant gets a random study ID; the key linking IDs to consent forms stays with the principal
  investigator on college premises, never with the dataset.
- Photos are re-encoded on intake to remove EXIF (location, device, time) and cropped to the mouth.
  Anything identifying (tattoos, jewellery with names, a face) is cropped or the photo is excluded.
- Storage: encrypted storage controlled by the institution. Not on personal laptops, not in git, not on
  public cloud storage, not uploaded to Kaggle as a public dataset. Training uses a private, access-
  controlled copy and is deleted from the training environment after each run.
- Labelling: boxes drawn by two dentists independently on the photos, guided by the clinical record;
  disagreements resolved by a third. Inter-rater agreement (Cohen's κ) is reported.
- Retention: as set by the committee (default proposal: 5 years after publication), then deletion.

## Use

- Split by participant (never by photo): 70 % train, 15 % validation, 15 % test. The test split is
  locked before any training run that uses this data.
- Report results separately for home photos and for AKU, with the same metrics as `EVAL.md`.
- Publications and the public app show only aggregate numbers, never participant photos, unless a
  participant gave separate, explicit consent for a specific image to be shown.
