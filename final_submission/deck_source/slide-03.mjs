import { base, bullet, C, FIG, figure, source } from "./common.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Brittleness", "The clean policy collapses under even mild post-training surprise.", "Frozen evaluation isolates the deployment-shift problem before any adaptation.");

  await figure(slide, ctx, FIG.degradation, 62, 190, 760, 420);
  ctx.addShape(slide, { x: 860, y: 202, width: 330, height: 392, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
  ctx.addText(slide, { x: 886, y: 230, width: 272, height: 54, text: "Why this matters", fontSize: 24, bold: true, color: C.ink, typeface: ctx.fonts.title });
  bullet(slide, ctx, "Clean reward is 1305.2, but mild surprise drops frozen reward to 105.9.", 890, 306, 258, C.red);
  bullet(slide, ctx, "Moderate/severe stay low; clean training does not transfer.", 890, 382, 258, C.orange);
  bullet(slide, ctx, "This motivates a post-training adaptation layer.", 890, 476, 258, C.blue);
  source(slide, ctx, "Source: runs/full_eval/evaluation_results.json");
  return slide;
}
