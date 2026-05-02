import { base, C, FIG, figure, metric, source } from "./common.mjs";

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Main result", "Lifelong adaptation helps in mild and severe, but moderate remains mixed.", "The honest story is partial robustness with clean retention, not a monotonic win.");

  await figure(slide, ctx, FIG.frozenVsLifelong, 62, 186, 710, 420);
  metric(slide, ctx, 820, 198, 170, "+4.1%", "clean", "1358.9 vs 1305.2", C.green);
  metric(slide, ctx, 1018, 198, 170, "+50.8%", "mild", "159.7 vs 105.9", C.green);
  metric(slide, ctx, 820, 320, 170, "-8.5%", "moderate", "45.2 vs 49.4", C.red);
  metric(slide, ctx, 1018, 320, 170, "+54.7%", "severe", "42.2 vs 27.3", C.green);
  ctx.addText(slide, {
    x: 820,
    y: 472,
    width: 368,
    height: 82,
    text: "Presentation phrasing: adaptation recovers some reward under surprise while preserving clean behavior in this seed.",
    fontSize: 17,
    bold: true,
    color: C.ink,
  });
  source(slide, ctx, "Source: runs/full_eval/evaluation_results.json");
  return slide;
}
