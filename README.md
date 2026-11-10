# limner

limner is a local image engine with one command, `lmn`. It does three jobs:

- **generate**: draws pictures on Apple silicon with
  [mflux](https://github.com/filipstrand/mflux) 0.22.0, Qwen-Image-Edit-2511 (4-bit) and the
  Lightning LoRA (8 steps). Each job edits a reference picture, or a blank canvas when it has none.
- **compose**: puts exact text, captions, speech bubbles and pictures on a canvas with Pillow, then
  checks the result and reports problems.
- **rasterize**: turns an SVG into a PNG with resvg, after it removes external references.

limner does not decide what to draw. The caller writes the prompts, the layout and the review steps
(the first caller is Herald). limner draws what the JSON says, keeps a record of how it made each
file, and returns JSON and an exit code. It never runs a shell and never writes outside `--root`.

## Install

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
# macOS on Apple silicon: compose, rasterize and generation
uv tool install "$HOME/projects/limner[generate]"    # keep the quotes in zsh

# anywhere else: compose and rasterize only
uv tool install "$HOME/projects/limner"

lmn fonts install      # pinned OFL fonts into ~/.cache/limner/fonts (needs the network once)
lmn setup              # Apple silicon only: makes the models ready (see "Model store")
lmn doctor --smoke     # one 64x64 picture with the real model
```

To work on limner itself, run `uv sync --extra generate`, then `uv run lmn …` and
`uv run --group dev pytest`.

## Commands

Every command prints one JSON line on stdout. `--result <file>` writes the same JSON to a file.

| command | what it does |
|---|---|
| `lmn run --job J --root R [--state-dir D] [--force ids] [--accept ids] [--wait N] [--check]` | Checks the job file. If there is work, it takes the locks, starts a detached worker and returns at once with `state: starting`. |
| `lmn status --run-id ID [--attempt A] --root R [--wait-seconds N]` | Reads the state of a run. Returns 10 while the run is still going. |
| `lmn cancel --run-id ID [--attempt A] --root R [--wait-seconds N]` | Asks the run to stop. If the worker is busy with a job, it stops the process group. |
| `lmn generate --job J --root R` | For people: `run`, then shows progress until the run ends. Ctrl-C cancels. |
| `lmn compose (--layout L \| --template T --data D) --root R --out O.png` | Draws text, bubbles and images, then checks them. |
| `lmn rasterize --svg S --root R --out O.png [--width N] [--background #fff]` | Converts an SVG to a PNG. |
| `lmn doctor [--json] [--smoke]` | Shows the platform, memory, model store, versions, fonts and the lock owner. |
| `lmn setup [--yes]` | Gets the required models (or finds them already there), then runs `fonts install`. |
| `lmn fonts install` | Downloads the pinned fonts and checks their sha256. |
| `lmn gc --state-dir D` | Lists what `prev/`, `runs/` and `tmp/` hold. It deletes nothing. |

### Exit codes

| code | meaning |
|---|---|
| 0 | success (`run` started, or found nothing to do) |
| 2 | bad input or environment (validation, paths, models, fonts, limits) |
| 4 | compose/rasterize wrote the file but found `problems` |
| 5 | a person must decide: an output was `modified`/`untracked` (use `--force` or `--accept`) |
| 10 | still `starting`/`running` |
| 11 | failed, crashed, cancelled, or `orphan_child` |
| 12 | `status`/`cancel` cannot find the run or attempt |
| 70 | internal error (a bug) |
| 73 | busy: another heavy job holds the machine |

### A job file

```json
{"version": 1, "model": "qwen-image-edit-2511",
 "jobs": [
   {"id": "ref-hero", "out": "out/ref-hero.png", "width": 640, "height": 800,
    "prompt": "…", "refs": [], "seed": 7},
   {"id": "p1", "out": "out/p1.png", "width": 896, "height": 896,
    "prompt": "…", "refs": ["out/ref-hero.png"], "seed": 7}]}
```

- Sizes are 64–1024 and must be multiples of 16. A job can have 0–2 references.
- When a reference is another job's `out`, that job runs first: jobs are sorted into dependency
  stages.
- References are letterboxed to the output's aspect ratio on `#E0E0E0` before generation.
- With two references, limner gives each one its own `Picture N:` slot in the vision-language
  prompt (`multi_ref: "picture_prefix"` in `limner.lock.json`), so both reach the encoder and the
  prompt can name them as image 1 and image 2. mflux 0.22.0 by itself passes only the first one.
- Guidance is 1.0, so limner skips the negative pass (`skip_negative: true`). The result is the
  same and the time is halved.

A run that finds an output already made with the same settings reports it as `skipped`. If a
person edited an output, the run reports it as `modified` and never overwrites it. When an output
is replaced, the old file is kept under `<state-dir>/prev/`. Records, old files and run state live
in the state dir (default `<root>/.limner/`, which callers should gitignore). They never go next to
the outputs.

Set `LIMNER_FAKE_GENERATOR=1` to replace the model with a solid-color generator for tests. It runs
on any platform. Its records say `"generator": "fake"`.

### compose

A layout has a canvas (`preset`: `instagram-4x5`, `instagram-1x1`, `og`, `x-16x9`, or
`width`/`height`) and layers of type `image`, `rect`, `text`, `caption` and `bubble`. Bubble kinds
are `speech`, `shout`, `thought` and `narration`. An optional `avoid` list gives boxes that text
must not cover, such as faces.

Text is NFC-normalized and laid out with Pillow's BASIC engine, so there is no kerning or shaping.
Lines break by the rules for Japanese, Korean and Latin text. limner draws the file, then reports
these problems:

`overflow`, `too_small`, `overlap`, `tail_crosses`, `off_canvas`, `missing_glyph`, `low_contrast`,
`unsupported_combining`

Built-in templates (`--template`): `og-title`, `card-cover`, `card-body`, `card-end`,
`strip-2x2` (2×2 panels) and `strip-3x2` (2 columns × 3 rows on 4:5). Each one has a
`<name>.schema.json`. `--templates DIR` is searched first.

## Model store, shared with script2video

limner uses the same model folders as [script2video](https://github.com/dev-yakuza/script2video),
so the 27 GB 4-bit model is stored only once:

- It uses the first of these that exists: `$LIMNER_MODELS_DIR`, `$S2V_MODELS_DIR`,
  `~/.cache/models`, `~/.cache/script2video/models`.
- Folders are named `<name>__<revision[:12]>`, with the same names and revisions as s2v's
  `imagegen.lock.json`.
- A folder counts as ready when it holds a `.complete` or `.s2v-complete` marker whose keys match.
  limner always writes both markers, and only into its own `<folder>.limner-partial/`. It never
  changes or deletes a folder another tool made, and it never deletes the source model.

If script2video has already made the models, `lmn setup` downloads nothing.

## Machine lock, shared with script2video and Studio

Heavy work on a 64 GB machine must not overlap. Generation, `setup` and `doctor --smoke` take
the lock at `$S2V_STATE_DIR/heavy.lock.d` (default `~/.s2v`). It uses the same directory-lock rules
and `owner.json` format as script2video and Studio. The code is a port of Studio's `dirlock.py`.

- limner holds the lock as `kind: "gen"`.
- Its holder string `limner run <run_id> <status.json>` tells a busy caller which run to check.
- When `lmn run` hands the lock to its worker, it gives the lock a new token, so a process that is
  breaking a dead lock never breaks the live worker's.
- limner is also busy while script2video's resident TTS server or a TTS/aligner server answers on
  ports 47861/47862.
- While a job runs, the worker watches the child's memory. The limit is `max(24, 0.6 × RAM)` GiB.

## Licensing

limner is MIT licensed (see `LICENSE`).

- `src/limner/compose/bubbles.py` is ported from
  [agent-comic-kit](https://github.com/Saeyeon-developer/agent-comic-kit) (Apache-2.0; see
  `LICENSES/agent-comic-kit-APACHE-2.0.txt` and the file header).
- `src/limner/dirlock.py` is ported from Studio, and the generation steps follow script2video
  `imagegen`. Both are by the same author.
- Fonts are downloaded, not bundled. Pretendard, Noto Sans JP, Noto Sans KR and JetBrains Mono are
  under the SIL Open Font License 1.1, and each one's OFL text is saved next to it.
- The models (Qwen-Image-Edit-2511, Qwen-Image-Edit-2511-Lightning) are Apache-2.0. Each record
  notes this in `model_license`.
