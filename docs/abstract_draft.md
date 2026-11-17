# Abstract draft

Target: EGU General Assembly 2027 (NH, natural hazards, snow avalanche session)
or ISSW 2028. Check the session's word limit; this draft is about 300 words.
Numbers come from results/multi_area.json and results/benchmark.json.

## Title

How far can an automated avalanche terrain map be trusted? Propagating input
uncertainty through AutoATES v2.0

## Abstract

Automated Avalanche Terrain Exposure Scale (ATES) mapping now covers large
regions, but each cell receives a single class with no indication of how
reliable it is. Where lidar is unavailable, maps depend on 30 m elevation
models and forest inventories whose errors are known to be large in steep,
forested terrain.

We reran the full AutoATES v2.0 chain (release areas, Flow-Py runout with forest
detrainment, classification) several hundred times per area, perturbing the
elevation model with spatially correlated error fitted against lidar, the
forest layer, and the classifier thresholds within the range of published
calibrations. Each cell then has a probability for each class. An open,
tested reimplementation runs the chain on a 10 m, 9 km2 grid in about 13 s.

We evaluated the probabilities in two ways. First, at five lidar-covered
blocks in British Columbia (four in the Coast Mountains, where no AutoATES
validation has been published), the error model was fitted on the other
blocks and the 30 m ensemble scored against a lidar-based run of the held-out
block. In the four Coast Mountains blocks a single 30 m map matched the lidar run
on 52 to 65% of cells; the ensemble halved the Brier score, and where more than
90% of runs agreed the single map matched on 75 to 92%. Second, at Connaught
Creek and Bow Summit we scored against published expert benchmark maps.
Agreement between runs again identified where the single map and the experts
differ (83% and 92% accuracy where runs agree, 59% and 63% where only half to
seven in ten do), but the probabilities were overconfident and did not
improve accuracy.

Forest input dominated the uncertainty in every configuration, even with lidar
elevation. Inventory crown closure and the national SCANFI product both
underestimate lidar canopy cover by 10 to 39 points, and a single scale factor
that transfers between blocks raises agreement with the lidar run from 56% to
67%, against 73% with lidar canopy itself. We propose
reporting a run-agreement layer with automated ATES maps to direct expert
review, and treating forest cover as the first input to improve.

## Notes before submitting

- Ask the AutoATES authors (Sykes, Toft) whether they want to be involved or
  know of overlapping work; credit them either way.
- EGU deadline listed by a third party as 12 January 2027; confirm on egu27.eu.
- ISSW 2028: Snowbird, Utah, dates to be announced on issw.net.
- Research only: the abstract should not suggest the maps be used for route
  decisions.
