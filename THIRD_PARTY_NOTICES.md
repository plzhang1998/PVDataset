# Third-party notices

This repository includes selected scripts/configurations and tools associated with:

1. **MMDetection / OpenMMLab** — <https://github.com/open-mmlab/mmdetection>
   Local source directory was named `mmdetection-3.2.0`; that name alone does not
   certify an unmodified upstream v3.2.0 checkout. See `third_party/MMDetection-LICENSE`.
   Existing source copyright headers are retained. Project edits parameterize
   paths and package a subset; model settings and geometry algorithms were not tuned.
2. **SAM 3 / Meta** — <https://github.com/facebookresearch/sam3>
   Only the project's custom evaluation entry point is included, not the framework
   or weights. `third_party/SAM3-LICENSE` preserves the licence text from the local
   installation for reference; it does not grant rights over all project additions.

Other runtime libraries retain their own licences. Exact framework commits and
the server's local modifications must be recorded before asserting reproducibility.
Author-written code licence remains undecided. Public availability is not a
declaration that every file is covered by Apache-2.0 or MIT.
