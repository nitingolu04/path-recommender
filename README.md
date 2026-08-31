# AI-Powered Personalised Learning Path Recommender

> Describe a learning goal in plain English. The app works out which skills that
> goal requires, subtracts what you already have, and builds the shortest credible
> sequence of courses, projects and assessments that closes the difference.
> Runs entirely locally. An LLM API key is **optional** and only widens the range
> of questions the assistant can answer — every recommendation is computed locally
> either way.

**Documentation map**

| File | Contents |
|---|---|
| [`BRIEF.md`](BRIEF.md) | The specification this implements — source of truth |
| [`SOLUTION.md`](SOLUTION.md) | Design: architecture, algorithms, calibration, measurements, limitations |
| `README.md` | Setup, running, and how the pieces fit together |

---

## 🚀 Quick Start

### 1. Clone / enter the project directory
```bash
cd learning-path-recommender
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note:** `torch` and `sentence-transformers` are the heavy packages (~1–2 GB).
> The first install may take a few minutes depending on your connection.

### 4. Run the app
```bash
streamlit run main.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
learning-path-recommender/
├── BRIEF.md                 # The hackathon brief — source of truth
├── .streamlit/
│   ├── config.toml          # Streamlit settings (committed)
│   └── secrets.toml.example # Template for the optional LLM key (copy, don't edit)
├── data/
│   ├── catalog.csv          # 129 resources: 84 courses, 25 projects, 20 assessments
│   └── recommender.db       # SQLite DB (auto-created on first run)
├── app/
│   ├── config.py            # Paths + cached catalog loading (single source of truth)
│   ├── db.py                # SQLite schema, migrations + CRUD helpers
│   ├── profiling_engine.py  # text → structured profile (rules + embeddings)
│   ├── learner_model.py     # stated preferences + observed learning patterns
│   ├── conversation.py      # message intent routing for multi-turn refinement
│   ├── skill_gap.py         # target skills, gap analysis, greedy set cover
│   ├── recommendation_engine.py  # MiniLM embeddings + bounded composite score
│   ├── path_generator.py    # topological sort + checkpoint insertion
│   ├── explainer.py         # type-aware explanations + template Q&A
│   ├── llm.py               # optional provider-agnostic LLM client
│   ├── ai_tutor.py          # grounded AI answers, templates as fallback
│   ├── pipeline.py          # the one place the end-to-end flow is expressed
│   ├── evaluation.py        # labelled goal set + TF-IDF baseline comparison
│   ├── charts.py            # Altair frame builders + chart specs
│   ├── chat_interface.py    # Chat tab
│   ├── dashboard.py         # Dashboard tab
│   └── eval_view.py         # "How It Performs" tab
├── main.py                  # Streamlit entrypoint (4-tab UI)
├── generate_catalog.py      # Regenerates data/catalog.csv (validates on write)
├── requirements.txt
├── SOLUTION.md              # Design documentation
└── README.md
```

---

## 🎯 What it does that a similarity search doesn't

The core of this app is **not** "embed the goal, return the nearest courses". That
approach fills its result quota regardless of whether a course teaches anything you
need — it put cloud-computing courses into a data-analyst path at 0.43 similarity.

Instead the app:

1. **Infers the skills your goal requires** and subtracts what you already have,
   producing an explicit gap.
2. **Selects the smallest set of courses covering that gap**, using greedy weighted
   set cover with "gap closed per study hour" as the objective. A course covering
   nothing you need is never selected, at any similarity.
3. **Sequences them** by prerequisite, inserting real projects and assessments at
   checkpoints once their dependencies are scheduled.
4. **Says so when it can't help.** A goal outside the catalog gets an honest refusal
   rather than five confident percentages.

On a beginner data-analyst goal this took the path from 14 loosely related steps to
4 courses and 24 hours covering 100% of the gap. See [`SOLUTION.md`](SOLUTION.md) §1
for the reasoning and §6 for the measurements.

---

## 🧠 How It Works

### 1. Goal Input (`chat_interface.py`)
The user types their learning goal in plain English. Example prompts are
provided as clickable pills. Streamlit session state persists the conversation
across interactions.

### 2. Profiling Engine (`profiling_engine.py`)
Converts raw text into a structured profile using a **hybrid approach**:
- **Clause-level skill extraction with polarity** — the text is split into
  clauses, each clause is tagged as describing something the user *has* or
  *wants*, and the 377-term skill vocabulary from the catalog is matched within
  each clause. This is what keeps "I want to learn React" out of
  `current_skills`. A clause with no cue of its own inherits the previous
  clause's polarity, so "I know Python, SQL and Excel" attributes all three.
- **Embedding-based domain classification** — embeds the goal text and computes
  cosine similarity against pre-computed embeddings of 5 category labels
  ("data science", "web development", etc.). Categories are kept only if they
  score within 72% of the best match, so a focused goal returns one or two
  domains rather than always exactly three.
- **Experience-level inference** — a priority ladder: an explicit
  self-declaration ("complete beginner", "no experience") wins outright, then an
  extracted years figure (>= 4 years is advanced, 1-3 intermediate), then
  seniority keywords. The years patterns tolerate words between the number and
  the noun, so "5 years of Python experience" is read correctly.
- The profile is **persisted to SQLite** keyed by `session_id`, with
  `current_skills` and `target_skills` stored separately.

### 3. Recommendation Engine (`recommendation_engine.py`)
- **Model:** `all-MiniLM-L6-v2` (sentence-transformers library)
  - 384-dimensional embeddings, optimised for semantic similarity tasks.
  - Runs fully locally — no API key needed, inference on CPU in < 1 s.
- **Indexing:** All course title+description pairs are embedded **once at startup**
  and cached as a `(N × 384)` numpy matrix. No per-request model calls for the
  catalog side.
- **Similarity metric:** Cosine similarity via dot-product on L2-normalised vectors.
  Scale-invariant, standard metric for sentence-embedding spaces.
- **Scoring:** The reported match is a bounded composite, not raw cosine:

  ```
  score = 0.75 * clamp(cosine, 0, 1) + 0.25 * level_fit
  ```

  `level_fit` is a 0-1 value keyed on the gap between the course's difficulty and
  the user's, so both terms are normalised and the composite is guaranteed to
  land in [0, 1]. Each component is returned alongside the composite
  (`semantic_score`, `level_fit`) and surfaced in the UI, so a score can always
  be decomposed into "how well it matches" versus "how well it fits your level".

### 4. Skill Gap Engine (`skill_gap.py`) — the core

The brief names skill-gap identification as one of the mechanisms the solution
works by. This is it, and it is what makes the app more than a search box.

- **Target skills** are inferred from the catalog, since no ground-truth "skills
  needed to be a data analyst" table exists. The goal is matched against *courses
  only* (a project exercises skills and an assessment checks them — neither
  teaches), and each skill's relevance is the similarity of the best course
  teaching it, which keeps it explainable: "SQL is a target because *SQL for Data
  Analysts* matches your goal at 0.79". Skills you name explicitly outrank
  anything inferred.
- **The gap** is that target set minus what you hold, counting both stated skills
  and skills acquired from resources you have completed.
- **Greedy weighted set cover** then selects the smallest set of courses covering
  the gap, maximising *gap closed per study hour* at each step. Set cover is
  NP-hard; greedy is the standard approximation within a `ln(n)+1` factor of
  optimal, and it is fast and explainable, which matter more here.
- **Skill equivalence is deliberately strict** — normalised exact match, not token
  subsets. Subset matching would treat "sql" as covering "sql injection", which is
  a different topic with identical token structure. Erring toward over-teaching
  beats telling someone they already know something they don't.
- **A servability floor** measured on your raw wording, not the enriched query.
  Below it the app says the catalog can't serve your goal instead of returning its
  least-bad guesses.

### 5. Path Generator (`path_generator.py`)
- **Topological sort** (Kahn's algorithm) over the prerequisite graph. Rather
  than a plain FIFO queue, the frontier of available resources is re-sorted each
  iteration by `(difficulty, -score)`, so the easiest best-matching available
  resource is always chosen next instead of an arbitrary valid one.
- **Bounded prerequisite expansion:** prerequisites the selector did not return
  are pulled in, but only 2 levels deep, and groundwork pitched below your level
  is skipped. Without both limits an advanced goal expanded into a 20+ resource
  path that opened with absolute basics.
- **Held skills discharge prerequisites.** Prerequisites are resource IDs, so
  nothing but `DS001` can satisfy a dependency on `DS001`. A resource whose skills
  are *all* already held is dropped, so five years of Python no longer earns you
  an introductory Python course. The test is strict, so partial overlap does not
  skip real groundwork.
- **Checkpoints insert real resources.** Every 3 resources, the sequencer adds an
  assessment and a project whose prerequisites are *all* already scheduled — which
  is what preserves topological validity by construction. Milestones used to be
  five fixed strings referring to project work that didn't exist as anything you
  could open; those remain only as a fallback.

### 6. Pipeline (`pipeline.py`)
The end-to-end flow — profile, analyse the gap, select for coverage, sequence,
trim to your pace, persist progress — lives in one module that every UI surface
calls. Keeping a single copy is what guarantees the dashboard's re-rank behaves
identically to the initial build; two copies is exactly how a missing
progress-row write once survived in one of them.

### 7. Conversation (`conversation.py`)
Messages are classified into one of seven intents before anything is applied, so
the chat is a conversation rather than a form submitted repeatedly. Only a
genuinely new objective rebuilds the profile — "I also know SQL" merges into what
you already told it.

Try: *"I also know Python"*, *"make it shorter"*, *"why this course?"*,
*"I've finished DS001"*, *"I only have 3 hours a week"*, *"start over"*.

Intent order is load-bearing: "I also want to learn Docker" contains "want to
learn", which alone reads as a brand-new goal, so the additive cue is tested
first. Classification is rule-based on purpose — the signal is a literal word like
"also" or "shorter", which is where patterns beat embeddings.

### 8. Learner Model (`learner_model.py`)
Stated **preferences** (format, pace, hours per week) are kept separate from
**observed patterns** (completion rate, per-type completion, category affinity),
because the two can disagree. Someone may say they prefer hands-on work and then
skip every project offered — the dashboard shows that contradiction rather than
averaging it away.

Each preference has a concrete effect: format decides whether a checkpoint opens
with a project or an assessment, hours drive the week-by-week schedule, and pace
bounds how much path is worth planning.

### 8b. Optional AI answers (`llm.py`, `ai_tutor.py`)

The **Ask a question** box under each step is answered by an LLM when one is
configured, and by the built-in templates when not.

**Setup is optional and takes one minute:**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# paste your key into LPR_LLM_API_KEY, then restart
```

Works with OpenAI, Groq, OpenRouter, Gemini, or a fully local Ollama / LM Studio
server — a key alone is enough, since the provider and model are inferred.
`secrets.toml` is gitignored.

**Why this is safe to add, when a path-generating LLM would not be.** The model
never selects a resource. Selection already happened in `skill_gap.py` from real
catalog data with real prerequisites. The model is handed a facts block — this
resource's description, skills, prerequisites, duration, which of your gap skills
it closes, your goal and level, and your full ordered path — and instructed to
answer only from it. It has nothing to fabricate, because it is never asked to
choose anything.

That distinction is the whole point: an LLM asked to *build* a learning path will
invent course names that don't exist. An LLM asked to *explain a course it was
given* cannot.

**It cannot break the app.** No key, dead network, rate limit, expired key or wrong
model name all fall back to the templates. The UI states which engine answered, so
an AI answer is never passed off as a computed one.

### 9. Explainer (`explainer.py`)
- **Gap coverage leads when available**: "closes 3 skills you're missing: SQL,
  joins, aggregations" is a stronger and more checkable reason than "scored 0.62
  against your goal".
- Templates distinguish held skills ("builds on your Python experience") from
  wanted skills ("teaches React, which you said you want to learn"), so a
  beginner is never credited with experience they don't have.
- **Type-aware**: a course *teaches* skills, a project *puts them into practice*,
  an assessment *checks your grasp of* them. Projects and assessments are explained
  by what unlocks them, since they were never scored — reporting a 0% match would be
  nonsense. Prerequisites are named by title, not as bare IDs.
- Q&A pattern-matches against 7 intents. Intents are ordered **most specific
  first** and the first match wins — ordering is load-bearing, since "What
  prerequisites do I need?" also contains the phrase "do I need".

### 10. Dashboard (`dashboard.py`, `charts.py`)
- **Skill gap section**: how many skills the goal needs, how many you have, what's
  still missing ranked by importance, and which gaps this path doesn't cover.
- **Skill development charts** (Altair, which ships with Streamlit): distinct skills
  acquired across your completion sequence, and where they came from by category.
  Only skills *new at that point* count, so an assessment covering material an
  earlier course already taught correctly flattens the curve instead of
  double-counting.
- **Schedule**: your path laid out week by week at your stated hours, with planned
  versus completed hours charted.
- **Learning patterns**: what you actually do, and a warning when it contradicts
  what you said you preferred.
- The two feedback buttons behave differently:
  - **Too Easy** removes the resource *and* counts toward promoting your
    difficulty level (every 2 reports raises it one tier, capped at advanced).
  - **Not Interested** removes it only, leaving difficulty alone.
- Both trigger a re-rank, which refreshes the persisted progress rows while
  preserving anything already marked complete.

### 11. Evaluation (`evaluation.py`, `eval_view.py`)
The **How It Performs** tab runs a real evaluation live rather than asserting
quality in prose. 33 labelled goals across all five domains, plus 7 goals the
catalog cannot serve, scored against a TF-IDF baseline over identical text with the
difficulty blend switched off on both sides.

| Ranker | P@1 | P@3 | P@5 | MRR | Must-have recall@10 |
|---|---|---|---|---|---|
| MiniLM embeddings | **1.00** | **0.90** | **0.82** | **1.00** | **1.00** |
| TF-IDF (lexical) | 0.94 | 0.85 | 0.69 | 0.96 | 0.94 |

The servability floor accepts 33/33 real goals and rejects 7/7 impossible ones.
End-to-end, five goals produce paths averaging 8 resources and 61 hours that close
90% of the identified gap, all prerequisite-valid.

The margin over TF-IDF is modest, and the tab says so: this catalog phrases things
much as learners phrase goals, which is where lexical matching competes well. The
tab also carries an explicit "what these numbers don't tell you" section.

---

## 🛠 Development Notes

### Regenerate the catalog
```bash
python generate_catalog.py
```
The generator validates before writing: no duplicate IDs, no dangling
prerequisite references, every resource has skills and a positive duration, and
every project and assessment is gated behind at least one prerequisite.

### Run module self-tests

Every module has a self-test that **asserts** its behaviour and exits non-zero on
failure, so these are usable as a regression suite. Each also carries explicit
regression assertions for bugs that were previously shipped.

```bash
python -m app.config                  # catalog loads, all three resource types present
python -m app.db                      # schema, migrations, progress, prior learning
python -m app.learner_model           # preferences, observed patterns, scheduling
python -m app.charts                  # chart frames incl. no double-counted skills
python -m app.skill_gap               # gap analysis, greedy cover, servability gate
python -m app.path_generator          # topological validity, checkpoint insertion
python -m app.explainer               # Q&A routing, type-aware explanations
python -m app.llm                     # config resolution, key redaction, failure handling
python -m app.ai_tutor                # facts grounding, template fallback, caching
python -m app.recommendation_engine   # ranking, score bounds (downloads model once)
python -m app.profiling_engine        # experience, skill polarity, prior learning
python -m app.conversation            # multi-turn intent routing and refinement
python -m app.pipeline                # end-to-end incl. progress persistence
python -m app.evaluation              # retrieval vs TF-IDF, gate, path quality
```

Run them all and report pass/fail (PowerShell):

```powershell
$mods = @('app.config','app.db','app.llm','app.learner_model','app.charts',
          'app.skill_gap','app.path_generator','app.explainer','app.ai_tutor',
          'app.recommendation_engine','app.profiling_engine','app.conversation',
          'app.pipeline','app.evaluation')
foreach ($m in $mods) {
  # Redirect stderr to its own file, not into the pipeline. Merging it with 2>&1
  # makes PowerShell treat the harmless HuggingFace warning as a command failure
  # and report a passing module as FAIL.
  python -m $m > "$env:TEMP\st_out.txt" 2> "$env:TEMP\st_err.txt"
  if ($LASTEXITCODE -eq 0) { "[PASS] $m" } else { "[FAIL] $m"; Get-Content "$env:TEMP\st_out.txt" -Tail 12 }
}
```

### Reset the database
```bash
rm data/recommender.db      # PowerShell: Remove-Item data\recommender.db
```
The DB is re-created automatically on the next `streamlit run`.

### Troubleshooting: `OSError: The paging file is too small` (Windows)

Torch reserves a large virtual address range when it loads the model. On a
Windows machine with a small or disabled page file this can fail even with free
physical RAM. Limiting thread count avoids it:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:TOKENIZERS_PARALLELISM = 'false'
streamlit run main.py
```

---

## 📊 Learning Catalog

129 synthetic resources across 5 categories and three types. The brief asks for a
roadmap of "courses, projects and assessments", so all three are first-class
entities in one table keyed by `resource_type`:

| Type | Role in a path | Count |
|---|---|---|
| `course` | Teaches skills | 84 |
| `project` | Applies skills taught elsewhere | 25 |
| `assessment` | Validates a cluster of skills | 20 |

By category:

| Category | Courses | Projects | Assessments |
|---|---|---|---|
| Data Science | 20 | 6 | 5 |
| Web Development | 17 | 5 | 4 |
| Cloud/DevOps | 17 | 4 | 3 |
| Business/Marketing | 15 | 5 | 4 |
| UX/UI Design | 15 | 5 | 4 |

By difficulty: 42 beginner, 57 intermediate, 30 advanced. Every prerequisite
reference resolves to a real ID, and every project and assessment is gated behind
the courses that teach its skills — so a project can never be scheduled before
the groundwork it depends on.

---

## 📝 Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Frontend | Streamlit | Fast to build, excellent for data apps |
| ML Model | `all-MiniLM-L6-v2` | Fast CPU inference, good semantic similarity quality |
| Database | SQLite (`sqlite3`) | Zero-config, file-based, perfect for prototypes |
| Data | pandas | CSV loading, filtering, manipulation |
| Similarity | numpy dot-product | Single matrix multiply for all N courses |

---

## 📄 License

MIT — use freely for learning and hackathon purposes.
