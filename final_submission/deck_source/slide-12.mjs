import { base, bullet, C, metric, source } from "./common.mjs";

export async function slide12(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Closing claim", "The system demonstrates clean retention and partial robustness.", "The remaining gap is stronger validation and a better adaptation objective.");

  ctx.addShape(slide, { x: 64, y: 190, width: 705, height: 232, fill: C.dark, line: ctx.line() });
  ctx.addText(slide, {
    x: 100,
    y: 238,
    width: 624,
    height: 118,
    text: "A clean-trained policy can be monitored for low confidence, adapted after deployment shift, and audited for forgetting.",
    fontSize: 30,
    bold: true,
    color: "#FFFFFF",
    typeface: ctx.fonts.title,
  });

  metric(slide, ctx, 818, 194, 170, "+50.8%", "mild reward", "lifelong vs frozen", C.green);
  metric(slide, ctx, 1018, 194, 170, "+54.7%", "severe reward", "lifelong vs frozen", C.green);
  metric(slide, ctx, 818, 318, 170, "false", "forgetting flag", "severe then clean", C.green);
  metric(slide, ctx, 1018, 318, 170, "seed 42", "scope", "single canonical seed", C.red);

  ctx.addShape(slide, { x: 64, y: 468, width: 1124, height: 150, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
  bullet(slide, ctx, "Novelty: integrated post-training surprise benchmark, confidence-triggered lifelong adaptation, and explicit clean-after-surprise forgetting audits.", 94, 500, 1000, C.blue);
  bullet(slide, ctx, "Future work: multi-seed validation, larger swarms, stronger confidence triggers, better adaptation objectives, and peer-help mechanisms.", 94, 558, 1000, C.orange);
  source(slide, ctx, "Source: canonical saved results and final report draft");
  return slide;
}
