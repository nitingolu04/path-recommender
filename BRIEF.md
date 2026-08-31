# Hackathon Brief — Source of Truth

> **This file is the only authoritative specification for this project.**
>
> The project was originally scaffolded from an AI-generated summary of this brief
> rather than from the brief itself. That summary turned out to be a lossy
> paraphrase: it dropped several requirements listed below and invented two that
> appear nowhere here (a FastAPI backend and a 150-200 row catalog target). It has
> since been removed to stop it being mistaken for the specification. Anything that
> disagrees with this file is wrong.

---

## AI-Powered Personalized Learning Path Recommender

Design and prototype an AI-powered solution that delivers personalized learning
experiences based on an individual's needs, interests, learning patterns and goals.

### Background

Online learning platforms offer thousands of courses across diverse domains.
While recommendation systems can suggest relevant courses, learners often
struggle to identify the right sequence of learning resources needed to achieve a
specific goal. Different learners have different skill levels, interests, career
aspirations and learning preferences, making a one-size-fits-all approach
ineffective. An AI-powered Personalized Learning Path Recommender can bridge this
gap by understanding a learner's profile, analyzing learning objectives,
identifying skill gaps and generating a structured roadmap of courses, projects
and assessments tailored to the individual.

### Task

Design and build an intelligent learning assistant that recommends personalized
learning paths based on a learner's interests, goals, previous learning history
and skill level. The solution should generate a structured learning roadmap,
explain recommendations, and adapt suggestions based on user feedback and progress.

### What to build

1. A conversational interface where learners describe their goals in natural language.
2. A learner profiling engine capturing interests, experience level, completed
   courses and objectives.
3. A recommendation engine suggesting relevant courses, projects and learning resources.
4. A personalized learning path generator with prerequisites and milestones.
5. An AI assistant that explains why each recommendation was made and answers
   learner queries.
6. A dashboard visualizing progress, skill development, milestones and next
   recommended actions.

---

## Judging Criteria

| Weight | Criterion |
|---|---|
| 20% | Problem Understanding & Solution Design |
| 25% | Functionality & Feature Completeness |
| 20% | AI/ML Implementation |
| 15% | Innovation & Creativity |
| 10% | User Experience & Interface |
| 10% | Performance & Code Quality |

**The final product must visibly demonstrate all six areas.** "Visibly" is
load-bearing: a mechanism that exists only in code, with no way for a judge to
see it during a demo, does not score.

---

## Requirements extracted from the prose

The "What to build" list is not the whole specification. The Background and Task
sections name mechanisms that the six bullets do not restate, and these are
requirements too:

| Source | Requirement |
|---|---|
| Title line | Personalize on needs, interests, **learning patterns** and goals |
| Background | Different learners have different **learning preferences** |
| Background | **Identify skill gaps** |
| Background | Roadmap of courses, **projects** and **assessments** |
| Task | Recommend based on **previous learning history** |
| Task | Adapt on user feedback **and progress** |

Every one of these was absent from the AI-generated summary the project was first
scaffolded from, which is why the initial implementation did not include them. They
are listed here separately because they are easy to miss: the "What to build" list
does not restate them, so reading only the bullets loses them.

---

## Explicit non-requirements

Listed here so they do not get re-added by mistake:

- **FastAPI / any HTTP backend.** Not mentioned anywhere in this brief; it came
  from the AI-generated summary. The recommendation flow is factored into
  `app/pipeline.py`, which contains no UI code, so an HTTP layer could be added
  later without restructuring. It would earn no marks against this brief, because
  it adds no learner-facing capability and the separation of logic from
  presentation it would demonstrate is already there.
- **A 150-200 row catalog.** Also invented by the summary. The brief sets no
  catalog size. What matters is that all three artifact types the brief names —
  courses, projects and assessments — are represented.

---

## Traceability

A table mapping each requirement above to the module that implements it and the
test that verifies it lives in `SOLUTION.md`.
