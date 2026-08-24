# Prompt Action Log

## 2026-06-26

### Prompt

User asked: "This repo is behind on some of the standard updates we've been making to other template repos like it needs the agents.md file and the prompt log and the ESIIL branding. Can you do a comprehensive update of the postdoc oasis so that it matches the working group oasis in style and setup but still geared toward postdocs instead of working groups. CU-ESIIL/Working_group_OASIS"

### Files and folders inspected

- Local Postdoc_OASIS repository structure
- Working_group_OASIS reference repository
- `mkdocs.yml`
- `README.md`
- `.github/workflows/`
- `docs/`
- `docker/`

### Actions taken

- Added `AGENTS.md` with postdoc-specific agent guidance.
- Added this prompt action log.
- Updated MkDocs navigation, branding paths, Material theme settings, CSS, and ESIIL visual styling.
- Reworked the homepage around the same repository-plus-website model used by Working_group_OASIS, adapted for postdoctoral researchers.
- Added postdoc work-plan and "how this postdoc project works" pages.
- Added ESIIL and team resource pages, public-facing site guidance, cloud triangle guidance, cite/reuse guidance, and practical GitHub/storage instructions.
- Added semantic image slots, process gallery folders, generated include files, and image slot generation scripts.
- Added non-blocking site health checks and a local template integrity check.
- Moved the JupyterLab runtime folder from `docker/` to `containers/` and updated the Docker image workflow.
- Updated GitHub Pages automation to generate image references, generate the site health report, and build with strict MkDocs before deploying.
- Added local artifact ignores for `.DS_Store`, `site/`, virtual environments, Node modules, and test results.
- Removed tracked `.DS_Store` files from the template.

### Verification

- Ran `python3 scripts/generate_image_slots.py`.
- Ran `python3 scripts/site_health.py`; remaining warnings are expected template placeholders.
- Ran `python3 scripts/check_template.py`.
- Ran `/private/tmp/postdoc_oasis_mkdocs_venv/bin/mkdocs build --strict --clean`.
- Confirmed strict build succeeds after fixing legacy missing asset links in older resource pages.

### Open questions and follow-up

- Confirm whether this template should keep the older resource guides in the main navigation or leave them available under the broader resources folder only.
