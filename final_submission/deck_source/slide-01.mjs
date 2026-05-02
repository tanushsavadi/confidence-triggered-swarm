import { base, C, metric } from "./common.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Final presentation | May 4, 2026", "Confidence-triggered lifelong adaptation for drone swarms", "A 10-12 minute graduate-level talk on clean-trained PPO, post-training surprise, confidence-triggered adaptation, and forgetting audits.");

  ctx.addShape(slide, { x: 54, y: 198, width: 1172, height: 250, fill: C.dark, line: ctx.line() });
  ctx.addText(slide, {
    x: 92,
    y: 236,
    width: 760,
    height: 122,
    text: "We train clean first, then test whether confidence-triggered adaptation can recover under surprise without forgetting clean flight.",
    fontSize: 34,
    bold: true,
    color: "#FFFFFF",
    typeface: ctx.fonts.title,
  });
  ctx.addText(slide, {
    x: 92,
    y: 368,
    width: 940,
    height: 36,
    text: "The claim is retention plus partial robustness, not a universal win over frozen PPO.",
    fontSize: 18,
    color: "#DDE7EF",
  });

  metric(slide, ctx, 78, 494, 250, "12 slides", "10-12 minute deck", "target talk time: 11 minutes", C.blue);
  metric(slide, ctx, 352, 494, 250, "Clean only", "baseline training", "no surprise during training", C.orange);
  metric(slide, ctx, 626, 494, 250, "8 figures", "report/deck artifacts", "generated from canonical JSONs", C.green);
  metric(slide, ctx, 900, 494, 250, "Seed 42", "canonical results", "multi-seed validation is future work", C.red);
  return slide;
}
