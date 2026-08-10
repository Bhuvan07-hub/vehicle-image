# Vehicle Image Processing Pipeline

A backend system that accepts uploaded vehicle images, queues them for
asynchronous analysis, and reports data-quality issues (blur, low light,
duplicates, screenshots, possible editing, invalid plate format, etc.) —
built for the Backend + AI Engineering take-home assignment.

**Stack:** Python 3.11+ · FastAPI · SQLite (stdlib `sqlite3`) · OpenCV ·
Pillow · Tesseract OCR · a custom in-memory asyncio queue.

---

## Table of Contents
- [Architecture](#architecture)
- [Analysis Checks](#analysis-checks)
- [AI Usage Disclosure](#ai-usage-disclosure-mandatory)
- [Trade-offs](#trade-offs)
- [Running Instructions](#running-instructions)
- [API Overview](#api-overview)
- [Assumptions](#assumptions)

---

## Architecture

### Service flow

```
                 ┌─────────────────────────────────────────────┐
                 │                  FastAPI app                 │
                 │                                                │
  Client ──POST──▶  /api/v1/images                                │
                 │     1. validate content-type & size            │
                 │     2. decode + verify image (Pillow)          │
                 │     3. compute sha256 + dHash (perceptual)      │
                 │     4. save file to disk (storage/<uuid>.<ext>) │
                 │     5. INSERT into `images` (status=pending)    │
                 │     6. enqueue(image_id)  ───────────┐          │
  Client ◀──202──   7. return {id, status, created_at}   │          │
                 │                                        │          │
                 │                              ┌─────────▼────────┐│
                 │                              │  asyncio.Queue    ││
                 │                              └─────────┬────────┘│
                 │                              N worker coroutines  │
                 │                              (asyncio.to_thread)  │
                 │                                        │          │
                 │                              ┌─────────▼────────┐│
                 │                              │  pipeline.py      ││
                 │                              │  run 8 checks     ││
                 │                              │  aggregate risk   ││
                 │                              │  UPDATE status +  ││
                 │                              │  INSERT result    ││
                 │                              └───────────────────┘│
                 │                                                    │
  Client ──GET───▶  /api/v1/images/{id}          (status)             │
  Client ──GET───▶  /api/v1/images/{id}/result   (analysis result)    │
  Client ──GET───▶  /api/v1/images               (list/filter)        │
                 └─────────────────────────────────────────────┘
```

The upload endpoint never blocks on analysis. It does the minimum synchronous
work needed to hand back a stable `id` (decode + hash + persist metadata),
then hands the actual CPU-heavy work (OpenCV, OCR) to the queue and returns
immediately with `202 Accepted`.

### Processing flow (per image)

1. `queue_worker` picks `image_id` off the `asyncio.Queue`.
2. Marks the DB row `processing`.
3. Runs `pipeline.process_image`, which:
   - Decodes the file with both Pillow (for EXIF/format-aware checks) and
     OpenCV (for pixel-level analysis).
   - Runs all 8 checks. **Each check is wrapped in its own try/except** —
     one check crashing doesn't fail the whole job; it's recorded as a
     `check_error` sub-result and the pipeline continues.
   - Aggregates `issues_detected` (list of triggered issue codes) and an
     `overall_risk_score` (the max confidence among triggered issues — see
     [Trade-offs](#trade-offs) for why not a weighted sum).
   - Writes the result row and marks the image `completed`.
4. If the image itself can't be decoded (corrupt upload) or an unexpected
   exception escapes step 3, the job is marked `failed` with an
   `error_message`, after `MAX_PROCESSING_RETRIES` bounded retries.

### Queue strategy

An **in-memory `asyncio.Queue`** with a small pool of worker coroutines
(default 2, configurable), each running the (CPU-bound, synchronous)
pipeline via `asyncio.to_thread` so it doesn't block the event loop that
also serves HTTP requests.

This was one of the explicitly-allowed options in the assignment brief. I
chose it over Redis/RabbitMQ/SQS because:
- **Zero extra infrastructure.** The whole thing runs with `pip install` +
  `uvicorn`, or one `docker compose up` — nothing else to stand up, no
  broker to configure, no extra container in the compose file.
- The reliability gap this creates (an in-flight job is lost if the process
  is hard-killed) is real, but partially mitigated:
  - **Job status lives in SQLite, not in the queue.** A client can always
    poll status regardless of queue state.
  - **Startup recovery**: on boot, the app re-enqueues every image still
    sitting in `pending`/`processing` from a previous run (see
    `recover_incomplete_jobs` in `app/queue_worker.py`), so a restart
    resumes stuck work instead of leaving it stuck forever.
- The queue is isolated behind two functions (`enqueue`, `start_workers`) —
  swapping in Redis/RabbitMQ/SQS later means rewriting one small module,
  not the rest of the system. See [Trade-offs](#trade-offs) for exactly
  what that migration would need to address.

### Major design decisions

| Decision | Reasoning |
|---|---|
| **SQLite via stdlib `sqlite3`, no ORM** | Two tables, no multi-database portability need for this assignment. An ORM (SQLAlchemy etc.) adds indirection without payoff here. Kept swappable behind a plain function-based `repository.py` — moving to Postgres later means changing `db.py`'s connection logic, not every call site. |
| **One JSON `checks` column per result, not one row per check** | Simpler to write and read atomically, and analysis results are read as a whole ("give me everything about image X"), not queried per-check across images. Documented as a trade-off below — this is the thing I'd change first if analytics-style querying became a requirement. |
| **Every check returns the same shape** (`check`, `issue_detected`, `issue_code`, `confidence`, `metrics`) | Lets the pipeline treat all 8 checks uniformly (aggregate, log, retry) without special-casing any one of them, and makes it trivial to add a 9th check later. |
| **`overall_risk_score` = max confidence among triggered issues** | A single high-confidence issue (e.g. an exact duplicate) should dominate the score, not get diluted by averaging in five unrelated "no issue found" checks contributing 0. |
| **Plate check reports `valid_format: false`, not "invalid plate"** | OCR failing to confirm a plate format is evidence for human review, not proof the vehicle's plate is actually wrong — see the checks section and trade-offs. Structuring around *uncertainty* rather than false certainty was one of the things this assignment explicitly asked to be evaluated on. |
| **Local disk storage, not S3** | No cloud credentials assumed for a take-home; `app/storage.py` is a 4-function module specifically so this is a one-file swap later. |

---

## Analysis Checks

All 8 checks live in `app/checks/`, one file each, and were individually
exercised against synthetic test images while building (see
[AI Usage Disclosure](#ai-usage-disclosure-mandatory) for how I validated
them — this wasn't just "write code and hope").

| Check | Method | Notes / known limitations |
|---|---|---|
| **Blur detection** | Variance of the Laplacian (OpenCV) | Cheap, no training data. Weakness: a flat, low-texture subject (e.g. a plain wall) can read as "low variance" without being blurry. Threshold is a single global number, not content-aware. |
| **Brightness analysis** | Mean grayscale intensity + fraction of near-black pixels | The dark-pixel-ratio signal catches "mostly dark frame with one bright headlight" cases that a pure mean would miss. |
| **Dimension validation** | Min width/height thresholds | Deterministic, not probabilistic — `confidence` is always 0 or 1. |
| **Duplicate detection** | SHA-256 (exact) + a hand-rolled dHash perceptual hash compared via Hamming distance against recent uploads | dHash implemented directly (no `imagehash` dependency) to keep the dependency list small. Compares against the most recent 5,000 uploads — see trade-offs on scale. |
| **Screenshot detection** | Weighted combination of: missing camera EXIF, screen-like aspect ratio, uniform top/bottom bands (status bar/home indicator), PNG format | None of these signals is individually reliable, so it's a weighted vote, not a single if/else. Explicitly a heuristic. |
| **Photo-of-photo detection** | FFT high-frequency energy ratio (moiré/interference pattern proxy) + border edge density (bezel detection) | The weakest of the 8 checks in practice — genuine recapture detection is a hard, actively-researched problem. Kept in because the assignment asked for it, but the README and the check's own docstring say so explicitly rather than overselling it. |
| **Editing/tamper heuristic** | EXIF `Software` tag match against known editors (Photoshop, GIMP, Snapseed...) + Error Level Analysis (ELA: re-compress at fixed JPEG quality, diff against original, look for localized hotspots) | ELA's docstring calls out its own biggest weakness: it's much more reliable on originally-uncompressed images, and most field photos already arrive as JPEG, so ELA alone is noisy. The EXIF signal is weighted higher for that reason. |
| **Plate format validation** | OpenCV contour-based plate-region localization → Tesseract OCR on the best candidate region → regex match against the standard Indian format (`^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$`) | This is the check I put the most "system design under uncertainty" thought into — see below. |

### Why plate OCR is structured the way it is

Running OCR on a whole vehicle photo (not a cropped plate) is close to
useless — too much irrelevant texture for Tesseract to key on. So this check
does actual localization first: Canny edge detection → contours → filter by
plate-like aspect ratio (2:1–6:1) and area → try OCR on the top candidates,
falling back to the full frame only if no candidate region is found at all.

I deliberately report `valid_format: false` as *"we could not confirm a
valid plate,"* not *"this plate is invalid."* Tesseract is a general-purpose
OCR engine, not a plate-specialized model — it will misread `0`/`O`, drop
characters on tilted or dirty plates, and fail outright on non-standard
formats (BH-series, older state formats). Treating a failed match as ground
truth would be actively wrong, and confidently wrong is worse than
uncertain. That's why the response carries a lower `confidence` (0.4) when
no plate region could even be localized (the OCR ran "blind" on the full
frame) versus 0.75 when a region was found but still didn't parse — the
system is trying to represent *how sure it is*, not just a binary verdict.

---

## AI Usage Disclosure (Mandatory)

I used Claude throughout this project. Concretely:

**Where AI helped:**
- Scaffolding the initial project structure (FastAPI app/router/repository
  split) and boilerplate (Dockerfile, `.gitignore`, Pydantic schemas).
- Drafting the first pass of each heuristic check (blur/brightness math,
  the dHash algorithm, the FFT-based photo-of-photo signal, the ELA diff
  logic, the plate-localization contour filter).
- Writing the pytest suite structure and the `seed.py` demo-image generator.
- Writing this README.

**Where AI output was wrong or needed correction, and how I validated it:**
- The first version of `screenshot.detect_photo_of_photo` leaked raw
  `numpy.bool_` / `numpy.float64` values into the result dict. This is a
  classic AI-generated-code trap: the code *looks* like it returns plain
  Python types, but comparisons against numpy arrays silently produce numpy
  scalar types. It passed a casual read-through but failed the moment it hit
  `json.dumps` — I caught this by actually running the check against a
  generated test image and serializing the output, not by reading the code.
  Fixed by explicitly casting with `bool()`/`float()`/`int()` at every
  numpy→Python boundary (see `app/checks/screenshot.py`).
- I did **not** trust the AI's first plate-regex attempt at face value — I
  independently checked it against real examples of the Indian plate format
  (`KA01AB1234`, `DL3CAF1234`) and deliberately-malformed ones (lowercase,
  wrong digit count) as unit tests, rather than assuming the regex was
  right because it looked plausible.
- I rejected the AI's first instinct to reach for `SQLAlchemy` +
  `imagehash` as dependencies. Given this environment couldn't reach the
  network to `pip install` anything not already present, I deliberately
  re-scoped to stdlib `sqlite3` and a hand-written dHash so that the parts
  of the system I could test in this sandbox (DB layer, all 8 checks, the
  full `pipeline.process_image` orchestration end-to-end, including a
  simulated corrupt-file failure path) actually got tested with real
  execution, not just written and assumed correct. This was a judgment call
  about validation depth over dependency "niceness."
- I could **not** install or run `fastapi`, `uvicorn`, or `pytest` in the
  sandbox I built this in (no network access). The API layer and the
  pytest test files were therefore written carefully and syntax-checked
  (`python -m py_compile`), but **not actually executed** end-to-end by me
  before handing this over. This is the single biggest honesty caveat in
  this disclosure: **run `pip install -r requirements.txt && pytest` and
  `uvicorn app.main:app --reload` yourself as the first real smoke test** —
  don't take the API layer's correctness on faith just because the rest of
  the system was verified.
- All 8 checks, the SQLite repository layer, and the full
  `pipeline.process_image` orchestration (including a deliberately-corrupted
  file to exercise the failure path) *were* executed against real
  synthetically-generated images during development — not just written.
  Where OCR came back with a plausible-but-imperfect read (e.g. reading
  `KA01AB1234` as `KAO1AB123`, confusing `0`/`O` and dropping a trailing
  digit), I left that result in rather than hand-tuning the test until it
  looked perfect — it's a realistic, honest demonstration of exactly the
  kind of OCR uncertainty this check is designed to surface, not hide.

**AI usage I'd flag as "used strategically, not blindly":** every check's
docstring stating its own weaknesses (blur threshold not being content-aware,
ELA being weak on pre-compressed JPEGs, photo-of-photo being the least
reliable check) was something I asked for and kept deliberately, rather than
letting the AI write the confident-sounding version. A model asked to "write
a blur detector" will happily hand back code with no caveats; getting it to
be honest about where the heuristic breaks down took explicit back-and-forth,
and that honesty is preserved in the shipped code, not just this README.

---

## Trade-offs

**What I intentionally simplified:**
- **No cloud storage** — files live on local disk. `app/storage.py` isolates
  this behind 3 functions so swapping to S3/GCS is a contained change, but
  it isn't done.
- **No normalized per-check table** — all 8 checks for one image live in a
  single JSON column. Fine for "fetch everything about image X" (the only
  read pattern this API needs), bad for "show me every image where
  `blur_detection` fired across the last 30 days" (would need a full table
  scan + JSON parse of every row). I'd split this into a proper
  `check_results(image_id, check_name, issue_detected, confidence, metrics)`
  table the moment analytics/dashboarding became a real requirement.
- **Duplicate detection scans the last 5,000 uploads per new image** — an
  O(n) linear scan of recent perceptual hashes. Fine at low volume; the real
  fix at scale is either (a) a proper perceptual-hash index (e.g. a
  BK-tree/VP-tree for Hamming-distance nearest-neighbor lookups, or a
  vector-DB-style approach), or (b) bucketing hashes and only comparing
  within a bucket.
- **Plate OCR uses general-purpose Tesseract**, not a plate-specialized
  model. Explicitly a "best-effort under uncertainty" implementation per
  the assignment's own framing, not a claim of production ANPR accuracy.
- **API auth**: none. Out of scope for the assignment; would add API-key or
  OAuth2 middleware before any real deployment.
- **No rate limiting** on the upload endpoint.

**What I'd improve with more time:**
- Per-check unit-test coverage against a curated set of *real* vehicle
  photos (blurry, dark, screenshotted, edited) instead of only synthetic
  ones — synthetic images validate the *code path*, but real-world photos
  would validate the *thresholds*, which currently are reasonable
  first-guesses, not tuned against labeled data.
- A confidence-calibration pass — right now each check's `confidence` is an
  ad-hoc formula (e.g. "distance from threshold, normalized"), not a
  calibrated probability. Fine for ranking/triage, not for anything that
  needs a true probability.
- Structured logging (JSON logs with request IDs) instead of the current
  plain-text `logging` module output, for easier log aggregation.
- A tiny status dashboard (the bonus points mention this) — would be a thin
  React/HTML page hitting the existing `/api/v1/images` list endpoint;
  didn't build it to keep scope focused on the core pipeline.

**Scalability concerns:**
- The in-memory queue means **this only works as a single process**. Running
  multiple app instances behind a load balancer would give you N independent
  queues, each unaware of the others — a real multi-instance deployment
  needs a shared broker (Redis/RabbitMQ/SQS), which is exactly the
  migration path `app/queue_worker.py` was written to make contained.
- SQLite's single-writer model is fine for this workload but is the first
  thing to swap (to Postgres) if concurrent write volume grows — the
  `repository.py` function signatures wouldn't need to change, only
  `db.py`'s connection/transaction handling.
- File storage on local disk doesn't scale horizontally or survive a single
  machine failure; same story, S3 is the real answer.

**Failure handling concerns:**
- The in-memory queue's biggest real gap: if the process is hard-killed
  *after* a check writes partial state but *before* the final status update,
  the startup-recovery logic will re-run that job from scratch (safe, since
  `save_analysis_result` is an upsert — see `ON CONFLICT` in
  `repository.py` — but wasteful). A production version would want
  idempotency keys or a proper "claimed but not yet acked" job state instead
  of just `pending`/`processing`/`completed`/`failed`.
- Retries are a flat bounded count (`MAX_PROCESSING_RETRIES`, default 2) with
  no backoff — fine for quick transient errors, not tuned for anything
  slower (e.g. a flaky external OCR API, if that check were swapped out for
  one).

---

## Running Instructions

### Option A — Docker (recommended, fewer moving parts)

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`. `storage/` and the
SQLite DB (`data/pipeline.db`) are mounted as volumes so data survives
container restarts.

Run the demo/seed script from your host once it's up:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install requests pillow numpy opencv-python-headless
python scripts/seed.py --base-url http://localhost:8000
```

### Option B — Local Python

Requires Python 3.11+ and the Tesseract OCR binary (`apt install
tesseract-ocr` on Debian/Ubuntu, `brew install tesseract` on macOS).

```bash
python3 -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env   # optional, defaults work out of the box

uvicorn app.main:app --reload
```

API docs (interactive Swagger UI): `http://localhost:8000/docs`

### Running tests

```bash
pytest
```

> **Note:** I built this in a network-isolated sandbox and could not
> `pip install` FastAPI/pytest there to execute these myself — see the
> [AI Usage Disclosure](#ai-usage-disclosure-mandatory) for exactly what
> was and wasn't actually run before this was handed over. Please treat
> `pytest` as your first real checkpoint, not a formality.

### Seeding sample data

```bash
python scripts/seed.py --base-url http://localhost:8000
```

Generates 8 synthetic images, each engineered to trip a specific check
(blurry, dark, overexposed, too-small, screenshot-shaped, an exact
duplicate, and a plate-bearing image), uploads them, and prints each
result once processing completes.

---

## API Overview

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/images` | Upload an image (multipart `file`). Returns `202` + `{id, status, created_at}`. |
| `GET` | `/api/v1/images/{id}` | Metadata + current status (`pending`/`processing`/`completed`/`failed`) + `error_message` if failed. |
| `GET` | `/api/v1/images/{id}/result` | Analysis result. `409` if not yet completed or if failed (with the failure reason). |
| `GET` | `/api/v1/images?status=&limit=&offset=` | Paginated list, optional status filter. |
| `GET` | `/health` | Liveness check. |

Full request/response examples with real curl commands:
[`docs/sample_requests.md`](docs/sample_requests.md)

---

## Assumptions

- "Vehicle images uploaded from the field" means phone-camera JPEGs/PNGs
  primarily, occasionally with WebP — hence the `ALLOWED_CONTENT_TYPES`
  allowlist rather than accepting arbitrary formats.
- Plate format validation targets the **standard Indian format**
  (`SS DD LLL DDDD`, e.g. `KA01AB1234`) as named in the assignment brief;
  BH-series and other newer/regional formats are out of scope but the regex
  is isolated in one constant (`plate_ocr.INDIAN_PLATE_REGEX`) if that needs
  extending later.
- A single-process deployment is acceptable for this assignment's scope (see
  [Trade-offs](#trade-offs) for what changes at multi-instance scale).
- "Reasonable" upload size cap set at 15MB — no size was specified, so I
  picked a number well above a typical phone photo (2–8MB) with headroom.
- No authentication was in scope; the API is assumed to sit behind whatever
  auth layer a real deployment would add.
