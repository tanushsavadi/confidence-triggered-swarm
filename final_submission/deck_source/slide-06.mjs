import { base, bullet, C, FIG, figure, metric, source } from "./common.mjs";

export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Forgetting check", "After severe adaptation, clean performance is retained in the saved run.", "This is the simplest clean-before versus clean-after probe.");

  await figure(slide, ctx, FIG.forgetting, 72, 186, 650, 410);
  metric(slide, ctx, 776, 204, 184, "1308.9", "baseline clean", "before severe adaptation", C.blue);
  metric(slide, ctx, 990, 204, 184, "1386.1", "post-adapt clean", "after severe adaptation", C.green);
  ctx.addShape(slide, { x: 778, y: 344, width: 396, height: 180, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
  ctx.addText(slide, { x: 806, y: 372, width: 330, height: 62, text: "Evaluator output: forgetting_detected = false", fontSize: 23, bold: true, color: C.green, typeface: ctx.fonts.title });
  bullet(slide, ctx, "Waypoints decrease slightly, so frame this as retention, not perfect improvement.", 804, 454, 322, C.orange);
  source(slide, ctx, "Source: runs/full_eval/evaluation_results.json");
  return slide;
}
