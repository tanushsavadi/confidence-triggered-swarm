import { base, bullet, C, label, metric, source } from "./common.mjs";

export async function slide11(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Work completed", "The final repo contains the implementation, experiments, audit trail, and submission artifacts.", "Use this slide to quickly cover what was done across the whole process before the closing claim.");

  const items = [
    ["Policy and environment", "Shared PPO/IPPO, FormationAviary task, surprise wrapper, clean baseline checkpoint."],
    ["Adaptation system", "Entropy/dropout confidence, replay buffers, reward-weighted updates, KL, clean replay, EWC."],
    ["Evaluation suite", "Frozen vs lifelong evaluation, severe ablations, forgetting checks, continual matrix."],
    ["Submission package", "NeurIPS-style report draft, generated figures, editable PPTX, speaker notes, readiness checker."],
  ];

  items.forEach((item, i) => {
    const x = 70 + (i % 2) * 575;
    const y = 194 + Math.floor(i / 2) * 178;
    ctx.addShape(slide, { x, y, width: 500, height: 132, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
    label(slide, ctx, item[0], x + 24, y + 22, 260, i % 2 === 0 ? C.blue : C.orange);
    ctx.addText(slide, { x: x + 24, y: y + 56, width: 438, height: 54, text: item[1], fontSize: 15, color: C.ink });
  });

  metric(slide, ctx, 126, 570, 210, "4 JSONs", "canonical result sources", "baseline/full/ablation/continual", C.blue);
  metric(slide, ctx, 382, 570, 210, "8 figs", "PNG + PDF outputs", "professor-ready", C.green);
  metric(slide, ctx, 638, 570, 210, "12 slides", "editable PPTX", "10-12 minute deck", C.orange);
  metric(slide, ctx, 894, 570, 210, "1 check", "readiness script", "flags manual items", C.red);
  source(slide, ctx, "Source: README.md, docs/artifact_guide.md, final_submission/");
  return slide;
}
