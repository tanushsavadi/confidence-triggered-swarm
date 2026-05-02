import { base, bullet, C, FIG, figure, source } from "./common.mjs";

export async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Continual evidence", "The matrix directly re-tests clean after later adaptation phases.", "Rows are completed training phases; columns are evaluation tasks.");

  await figure(slide, ctx, FIG.continualMatrix, 54, 208, 850, 390);
  ctx.addShape(slide, { x: 930, y: 208, width: 274, height: 390, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
  ctx.addText(slide, { x: 954, y: 234, width: 226, height: 56, text: "Clean retention column", fontSize: 24, bold: true, color: C.ink, typeface: ctx.fonts.title });
  bullet(slide, ctx, "After clean: 1316.0", 956, 318, 210, C.blue);
  bullet(slide, ctx, "After mild: 1261.3", 956, 368, 210, C.orange);
  bullet(slide, ctx, "After moderate: 1406.2", 956, 418, 210, C.green);
  bullet(slide, ctx, "After severe: 1322.9", 956, 468, 210, C.green);
  ctx.addText(slide, { x: 956, y: 544, width: 210, height: 34, text: "Use this slide to answer: did we go back and test clean?", fontSize: 14, bold: true, color: C.red });
  source(slide, ctx, "Source: runs/continual_run/continual_results.json");
  return slide;
}
