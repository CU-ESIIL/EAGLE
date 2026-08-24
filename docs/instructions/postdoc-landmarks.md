# Postdoc Project Landmarks

Postdoc project landmarks are lightweight labels that connect instruction pages to the homepage sections postdocs and collaborators may need to edit. They are not a new workflow and they are not a visual sticker system. They are simple markers for finding the right part of the site.

## Labels

| Label | Meaning | Common homepage or repo location |
| --- | --- | --- |
| PD-A | People and roles | Team Members, community expectations, meeting notes |
| PD-B | Question and scope | Homepage title, Start Here, Current Phase, Work Plan |
| PD-C | Data and access | Resources, Cloud Triangle notes, persistent storage README files |
| PD-D | Methods and workflows | Repository Side, Work Plan, scripts, notebooks, workflow galleries |
| PD-E | Results and synthesis | Website Side, outputs galleries, synthesis notes |
| PD-F | Outputs and handoff | Outputs and Wrap Up, public site guide, cite and reuse notes |

Use the labels when meeting notes or instruction pages need to point postdocs and collaborators toward the place where a change belongs.

## How To Use The Labels

When a postdoc or collaborator asks "Where should this go?", choose the nearest landmark:

- People, roles, responsibilities, or norms: PD-A.
- Research question, scope, purpose, or audience: PD-B.
- Dataset access, metadata, storage paths, or data provenance: PD-C.
- Scripts, notebooks, computational steps, or reproducibility notes: PD-D.
- Findings, uncertainty, figures, or interpretation: PD-E.
- Final outputs, reuse instructions, citations, or archiving: PD-F.

Add the label in plain Markdown near the relevant note. For example:

```markdown
Landmark: PD-C Data and access
```

or:

```markdown
Related landmarks: PD-D Methods and workflows; PD-E Results and synthesis
```

## Homepage Connections

The current homepage already has natural places for these labels:

- Team Members: PD-A People and roles.
- Start Here and Current Phase: PD-B Question and scope.
- Repository Side and Resources: PD-C Data and access; PD-D Methods and workflows.
- Website Side and Outputs sections: PD-E Results and synthesis; PD-F Outputs and handoff.
- Work Plan: all landmarks, depending on the phase.

Keep labels subtle. A short line is enough. Do not clutter the homepage with repeated badges.

Future visual design note: if the postdoc project later wants sticker-style visual markers, add them as a small CSS and image-system improvement rather than embedding complex HTML in Markdown.

## Related Pages

- [Postdoc Project Lifecycle](postdoc-project-lifecycle.md)
- [Public-Facing Site Guide](../public-facing-site-guide.md)
