# EAGLE: Extracting Airborne Gestalts from Lidar for Ecology

## How this repository is organized

The repository has two connected layers. Top-level files configure the project and its automation. The `docs/` folder contains the website content. `mkdocs.yml` tells MkDocs how to turn that content into the public site. Analysis folders hold the working scientific materials that generate the results shown on the website.

```text
.
├── README.md              # Repository overview and setup notes
├── AGENTS.md              # Guidance for coding agents and future maintainers
├── PROMPT_ACTION_LOG.md   # Record of template-level prompt-driven changes
├── mkdocs.yml             # Website navigation, theme, plugins, and edit links
├── docs/                  # Markdown source for the public website
├── scripts/               # Data processing, ingestion, and analysis scripts
├── src/                   # Re-usable code for analyses
└── datasets/              # Curated and cleaned datasets for LiDAR
```


## Preview locally

```bash
pip install -r requirements.txt
python scripts/generate_image_slots.py
python scripts/site_health.py
mkdocs serve
```

## Build site

```bash
python scripts/generate_image_slots.py
python scripts/site_health.py
mkdocs build --strict --clean
```

## Swapping homepage images

The site uses semantic image slots so postdocs do not need to edit Markdown links every time an image changes.

1. Open the relevant folder in `docs/assets/images/slots/`.
2. Delete the old image file.
3. Add one new `.png`, `.jpg`, `.jpeg`, `.webp`, or `.svg` file.
4. Run `python scripts/generate_image_slots.py`.
5. Commit the image change and the regenerated slot references.

If a slot folder contains multiple images, the generator uses the first image alphabetically and the site health report will warn you to clean it up. The cleanest workflow is still one image per slot folder.

## Using process galleries

Process galleries are folder-driven. Add files to a gallery folder, commit them, and the site updates automatically.

1. Open the relevant folder in `docs/assets/images/process/`.
2. Add images or supported deliverable files.
3. Optionally add a `captions.txt` file with lines like `filename.png | Caption text`.
4. Run `python scripts/generate_image_slots.py`.
5. Commit the new files and the regenerated gallery includes.

Supported image files:

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`
- `.svg`

Supported linked deliverable files:

- `.pdf`
- `.html`
- `.csv`
- `.xlsx`
- `.docx`
- `.pptx`

## Site Health

The site generates a non-blocking health report during the build.

The report flags common issues such as missing files, placeholder links, outdated navigation, or incomplete template fields.

Warnings do not prevent the site from publishing. They are intended to help postdocs and maintainers improve the site.