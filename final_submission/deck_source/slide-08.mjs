import { base, bullet, C, FIG, figure, metric, source } from "./common.mjs";

export async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Ablation evidence", "The safeguards change the severe-surprise tradeoff, but the trigger is still conservative.", "This is the graduate-level caveat: the method has mechanisms for stability, but adaptation frequency and data quality still bottleneck the result.");

  await figure(slide, ctx, FIG.ablations, 62, 188, 720, 406);
  metric(slide, ctx, 824, 204, 170, "0", "full-method adapts", "15 severe episodes", C.orange);
  metric(slide, ctx, 1024, 204, 170, "1", "ablated adapts", "looser behavior appears", C.blue);

  ctx.addShape(slide, { x: 824, y: 344, width: 370, height: 210, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
  bullet(slide, ctx, "Ablations can raise severe reward but relax clean guardrails.", 850, 374, 292, C.blue);
  bullet(slide, ctx, "Improve triggers and objectives before deleting safeguards.", 850, 470, 292, C.red);
  source(slide, ctx, "Source: runs/ablations/ablation_results.json");
  return slide;
}
