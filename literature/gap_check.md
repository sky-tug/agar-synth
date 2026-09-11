# Research gap verification — first pass

> 13 August 2026. The gap statement had been drafted, with a note attached:
> *"only after a literature check confirms it"*. This file is that check,
> first round.
>
> **Result: no identical study found.** But the gap sentence must not be
> written before three papers are read in full — marked below.

---

## The gap we claim

Three components that have not been done **together**:

1. **Domain:** colony **detection** on agar plates (counting-oriented, not
   segmentation)
2. **Method:** mask-controlled diffusion inpainting → the label exists before
   generation, at zero cost
3. **Question:** a quantitative measurement of **how much** synthetic data
   substitutes for real — nested subsamples, control arms, a substitution curve

Each one exists in the literature separately. No intersection of the three was
found.

---

## 🔴 The gap sentence is not written before these three are read

| # | Study | Why it is a threat | What to check |
|---|---|---|---|
| 1 | **Denoising diffusion probabilistic models for generation of realistic fully-annotated microscopy image datasets** — PLOS Comput Biol, 2024 | Methodologically the closest: diffusion plus **fully annotated** microscopy dataset generation | How is the label produced — mask first, as here? Does it measure a substitution rate, or only claim "synthetic data helps"? Detection or segmentation? |
| 2 | **Deep generative modeling of annotated bacterial biofilm images** — npj Biofilms and Microbiomes, 2025 | **Bacteria** + **annotated** + generative. Closest in domain | Biofilm or colony (different imaging, different problem)? Diffusion or GAN? Is there a downstream detection task? |
| 3 | **Diffusion-Based Synthetic Brightfield Microscopy Images for Enhanced Single Cell Detection** — arXiv:2512.00078 | Diffusion + microscopy + **detection**. Two of the three components | Single cell versus colony. Does it draw a substitution curve? Are there control arms? |

None of the three appears to close the gap — none combines agar plates with a
substitution curve — but that cannot be settled without reading them.

---

## 🟡 Direct predecessor — the introduction must position against this

**Generation of microbial colonies dataset with deep learning style transfer**
— Scientific Reports, 2022

The AGAR team's **own** work. Same dataset, same motivation, but via **style
transfer**. This is exactly the alternative the project proposal dismissed as
"already tried on this dataset".

It is both threat and opportunity:

- **Threat:** the source of any "this has already been done" objection
- **Opportunity:** a direct comparison point. They used style transfer, we use
  diffusion. How did they obtain labels? Did they measure a substitution rate?
  If they did not, the contribution is clean: *the first study to ask that
  question quantitatively on this dataset.*

---

## 🟢 Method ancestors — not a problem, to be cited

| Study | Relation |
|---|---|
| **Outline-Guided Object Inpainting with Diffusion Models** — arXiv:2402.16421 | The general-domain ancestor of the method: the object outline is given in advance and inpainting fills it, so the label is free. Our difference: microbiology domain, LoRA domain adaptation, substitution curve |
| **SmartBrush: Text and Shape Guided Object Inpainting** — CVPR 2023 | Shape-controlled inpainting, foundational reference |
| **Exploring the potential of synthetic data to replace real data** — arXiv:2408.14559 | The general-domain version of the substitution question |
| **The Impact of Synthetic Data on Object Detection Model Performance** — arXiv:2510.12208 | Reference for the synthetic/real comparison protocol |

---

## 🟢 Same dataset, no overlap

| Study | Why it does not overlap |
|---|---|
| **AGAR: a microbial colony dataset for deep learning detection** — arXiv:2108.01234 | The dataset itself. Our baseline reference (lower-resolution mAP 59.4%) |
| **Colony Grounded SAM2** (2025) | AGAR with Grounding DINO / SAM2, **no synthetic generation**, entirely real data |
| **Assessing microbial colony counting: a deep learning approach with the AGAR image dataset** — Neurocomputing, 2025 | Counting-oriented, real data. Reference for our counting metrics (MAE / sMAPE) |
| **Bacterial Colony Counting and Classification System Based on Deep Learning** — Applied Sciences, 2026 | Recent counting work, reference |

---

## Next steps

1. Import the twelve records above into the reference library
2. Read the three 🔴 papers and add a paragraph of notes each, below
3. Read the Sci Rep 2022 style-transfer paper **carefully** — the introduction
   will be written against it
4. Write the gap sentence and send it for supervisor approval
5. Target 30+ references, distributed across the four subsections of the
   related-work chapter (diffusion · training on synthetic data · microbial
   colony detection · learning from few labels)

---

## Caveat

This pass is based on search results; the full texts were not read. **The gap
sentence cannot be written on the strength of this file** — it is written after
the three 🔴 papers are read.
