import { base, bullet, C, FIG, figure, metric, source } from "./common.mjs";

export async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Training over time", "The sequential run exposes where adaptation helps and where it remains noisy.", "The same lifelong policy moves through clean, mild, moderate, and severe phases, then is re-evaluated after each phase.");

  await figure(slide, ctx, FIG.trainingOverTime, 62, 188, 735, 404);
  metric(slide, ctx, 838, 198, 170, "clean", "phase 1", "high-reward behavior", C.green);
  metric(slide, ctx, 1030, 198, 170, "severe", "phase 4", "still low and noisy", C.red);

  ctx.addShape(slide, { x: 836, y: 328, width: 366, height: 218, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
  bullet(slide, ctx, "The work includes per-episode traces plus after-phase matrix evaluation.", 862, 358, 292, C.blue);
  bullet(slide, ctx, "Severe remains difficult even when adaptation recovers some reward.", 862, 452, 292, C.red);
  source(slide, ctx, "Source: runs/continual_run/continual_results.json");
  return slide;
}
