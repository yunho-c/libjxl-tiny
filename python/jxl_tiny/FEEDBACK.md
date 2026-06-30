### 6/28/2026

- Let's use `ruff`, so that we have deterministic formatting
  - Using 2 spaces was a mistake; let's convert to 4-spaces indent when we can
- In file header docstring, let's explain the purpose of the module / what it does
  - in addition to just stating that it mirrors a certain file from `libjxl-tiny` C++ impl
- Let's put function docstrings / line comments where applicable / beneficial, to help future readers (to help them understand JPEG-XL encoding process as much as possible!)

### 6/30/2026

- [x] Add a reader walkthrough that follows one tiny fixture from pixels to codestream.
  - Implemented in `doc/python-walkthrough.md`.
- [x] Add a compact glossary for core Python-port/JPEG XL terms.
  - Implemented in `doc/python-port.md`.
- [x] Add a trace-inspection recipe using real trace artifact names.
  - Implemented in `doc/python-walkthrough.md`.
- [x] Add a JPEG-to-JPEG-XL mental model for readers familiar with baseline JPEG.
  - Implemented in `doc/python-port.md` and `doc/python-walkthrough.md`.
