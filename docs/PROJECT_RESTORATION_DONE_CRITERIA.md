# Project Restoration Done Criteria

## Scope

This document defines what should be considered "done" for the restoration of the lyric word-timestamp and chunking project.

The goal is not to claim that every accuracy problem has been solved. The goal is to restore the project to a state that is usable, reproducible, inspectable, and structurally clear.


## Level 1: Minimum Deliverable

The restoration task can be considered minimally complete when all of the following are true:

1. The project can reliably take `audio + LRC` as input.
2. The project can reliably output:
   - word-level timestamps (`start`, `end`, `text`)
   - chunk-level results (`start`, `end`, `text`)
3. The main scripts, dependencies, parameters, and artifact paths are documented.
4. The key visualization workflow can be reproduced:
   - single-preview video
   - 2x2 comparison video
   - audio mux
   - timer overlay
   - chunk display
   - active-word highlight
5. The full pipeline can be run once from a clean environment using a documented SOP.

If these conditions are satisfied, the project is "restored and usable".


## Level 2: Proper Closure

The restoration task should only be considered fully closed when all of the following are true:

1. The codebase structure is stable and layered clearly:
   - alignment
   - timing refinement
   - chunking
   - feature extraction
   - video visualization
   - documentation
2. Artifact management is stable:
   - naming rules are consistent
   - final artifacts and intermediate artifacts are clearly separated
3. The parameter system is understandable:
   - default parameters are documented
   - the effect of major parameters on word starts, word durations, and chunk boundaries is explained
4. Accuracy is validated against an explicit standard:
   - not just "looks acceptable"
   - but checked against human anchors or a manual inspection set
   - with a defined acceptable error range
5. Basic generalization has been checked:
   - not only on one song
   - but on several songs with different rhythm and vocal conditions
6. Missing historical details have been resolved properly:
   - anything that can be restored has been restored
   - anything that cannot be restored exactly has been converted into a requirements document instead of being guessed


## Recommended Definition of Done for This Project

For this specific project, the restoration task should be treated as complete only when the following six conditions are all satisfied:

1. The word-timestamp pipeline is restored and the default strategy is explicit.
2. The chunking pipeline is restored and produces stable output.
3. The video module is restored and usable for human inspection.
4. The documentation is complete enough for another person to reproduce the full pipeline.
5. A clear validation mechanism exists, such as manual anchor checks.
6. All uncertain historical behavior has either been restored or written down as formal requirements.


## Important Clarification

"Done" does not mean "perfect".

This restoration task should not wait for every timing error to be eliminated. A project can be considered complete even if some known issues remain, as long as:

1. the main pipeline is complete,
2. the structure is clear,
3. the workflow is reproducible,
4. the outputs are verifiable,
5. and known problems are explicitly documented.


## Known Issues Can Still Exist at Completion

The following types of issues may still remain when the restoration task is considered complete:

1. some short words start slightly too early,
2. some long words are stretched longer than perceived,
3. singing-domain generalization is still limited,
4. some visual details of historical preview modules are approximate rather than pixel-identical.

These do not block project closure if they are documented and bounded.


## Final Acceptance Checklist

The restoration task is complete when all items below are true:

1. The full pipeline can be reproduced from a clean environment to final JSON and preview videos.
2. Default and optional parameters are documented.
3. Chunk display and word highlighting are logically correct in the preview video.
4. At least one human-anchor validation workflow exists.
5. Final-file naming and directory rules are fixed and documented.
6. Any historically uncertain behavior has been moved into explicit requirement documents.
