---
name: humanize-docs
description: Transform rigid, AI-generated, or overly formal technical documentation into natural, readable prose while preserving technical accuracy. Use when editing READMEs, docs, tutorials, specs, changelogs, reports, or internal documentation for clearer human tone.
---

# Humanize Docs

Use this skill when the user asks to humanize, de-formalize, polish, or remove robotic phrasing from documentation. The goal is not to hide authorship or change meaning. The goal is to make technical writing easier to read, more direct, and more useful.

## Core Rules

- Preserve every technical claim, command, API name, path, variable, metric, result, and caveat unless the user explicitly asks for content changes.
- Do not invent context, benefits, benchmarks, limitations, citations, or user stories.
- Keep the document's purpose and audience intact. A README can be friendly; a technical spec should stay precise.
- Prefer plain words over corporate filler: use "use" instead of "utilize", "help" instead of "facilitate", and "show" instead of "demonstrate" when the simpler word fits.
- Replace repetitive list scaffolding with prose only when it improves scanning. Keep checklists, requirements, or step-by-step procedures when structure matters.
- Vary sentence length and rhythm, but do not add jokes, slang, or personality that would feel out of place in the project.
- Keep headings, code blocks, tables, frontmatter, links, and structured data valid.

## Workflow

1. Identify the document type: README, tutorial, API docs, internal notes, spec, report, changelog, or blog-style writing.
2. Infer the target tone from surrounding project docs unless the user specifies one.
3. Scan for common robotic patterns:
   - symmetrical sections that all start and end the same way
   - padded transitions such as "Moreover", "Furthermore", "It is important to note"
   - inflated claims such as "robust", "seamless", "cutting-edge", or "powerful" without evidence
   - excessive bullets where a short paragraph would read better
   - repeated sentence openings and conclusions that summarize the obvious
4. Rewrite for directness first, then rhythm.
5. Re-check that commands, values, filenames, citations, and constraints are unchanged.
6. If editing a file, keep the diff focused on prose. Avoid unrelated formatting churn.

## Tone Calibration

- READMEs: clear, helpful, practical. A little warmth is fine.
- Tutorials: conversational and encouraging, but still concise.
- API docs: direct and predictable. Do not sacrifice precision for personality.
- Specs and reports: professional, specific, and readable. Keep uncertainty explicit.
- Changelogs: concrete and user-facing. Avoid marketing language.

## Output Expectations

When returning edited text in chat, provide the polished version only unless the user asks for commentary. When editing files, summarize the style changes briefly and mention any technical ambiguity you preserved rather than resolving.
