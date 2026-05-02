import { base, bullet, C, FIG, figure, metric, source } from "./common.mjs";

export async function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Continual-learning metrics", "The result is retention and transfer, not a final-average reward win.", "This is the cleanest way to connect the project to lifelong learning and catastrophic-forgetting evaluation.");

  await figure(slide, ctx, FIG.clMetrics, 64, 184, 610, 390);
  metric(slide, ctx, 728, 190, 170, "393.1", "lifelong final avg", "below frozen 412.9", C.orange);
  metric(slide, ctx, 930, 190, 170, "+15.4", "BWT", "positive in seed 42", C.green);
  metric(slide, ctx, 728, 310, 170, "+17.2", "FWT", "positive in seed 42", C.green);
  metric(slide, ctx, 930, 310, 170, "1.0", "remembering", "no clean forgetting", C.green);

  ctx.addShape(slide, { x: 720, y: 446, width: 410, height: 168, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
  bullet(slide, ctx, "Backward/forward transfer are positive in this seed.", 744, 474, 340, C.green);
  bullet(slide, ctx, "Final average is lower, so the honest claim stays limited.", 744, 532, 340, C.orange);
  source(slide, ctx, "Source: runs/continual_run/continual_results.json");
  return slide;
}
