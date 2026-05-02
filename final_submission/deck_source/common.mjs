import { fileURLToPath } from "node:url";
import path from "node:path";

export const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

export const FIG = {
  frozenVsLifelong: path.join(ROOT, "runs/professor_ready/fig1_frozen_vs_lifelong.png"),
  degradation: path.join(ROOT, "runs/professor_ready/fig2_degradation.png"),
  ablations: path.join(ROOT, "runs/professor_ready/fig3_ablations.png"),
  forgetting: path.join(ROOT, "runs/professor_ready/fig4_forgetting.png"),
  trainingOverTime: path.join(ROOT, "runs/professor_ready/fig5_training_over_time.png"),
  continualMatrix: path.join(ROOT, "runs/professor_ready/fig6_continual_matrix.png"),
  cleanRetention: path.join(ROOT, "runs/professor_ready/fig7_clean_retention.png"),
  clMetrics: path.join(ROOT, "runs/professor_ready/fig8_cl_metrics.png"),
};

export const C = {
  bg: "#F7F4EE",
  paper: "#FFFFFF",
  ink: "#1D242B",
  muted: "#61717E",
  rule: "#D8D1C4",
  blue: "#2F6DB5",
  orange: "#D77A2D",
  green: "#2E7D55",
  red: "#A9433C",
  yellow: "#D6A63A",
  dark: "#22313D",
};

export function base(slide, ctx, kicker, title, subtitle = "") {
  ctx.addShape(slide, {
    x: 0,
    y: 0,
    width: ctx.W,
    height: ctx.H,
    fill: C.bg,
    line: ctx.line(),
  });
  ctx.addText(slide, {
    x: 54,
    y: 34,
    width: 240,
    height: 24,
    text: kicker.toUpperCase(),
    fontSize: 13,
    bold: true,
    color: C.orange,
    typeface: ctx.fonts.body,
  });
  ctx.addText(slide, {
    x: 54,
    y: 62,
    width: 810,
    height: 72,
    text: title,
    fontSize: 30,
    bold: true,
    color: C.ink,
    typeface: ctx.fonts.title,
  });
  if (subtitle) {
    ctx.addText(slide, {
      x: 54,
      y: 150,
      width: 790,
      height: 42,
      text: subtitle,
      fontSize: 15,
      color: C.muted,
      typeface: ctx.fonts.body,
    });
  }
  ctx.addShape(slide, {
    x: 54,
    y: 670,
    width: 1172,
    height: 1.5,
    fill: C.rule,
    line: ctx.line(),
  });
  ctx.addText(slide, {
    x: 54,
    y: 681,
    width: 640,
    height: 18,
    text: "CS 590NN/690NN final project | Confidence-triggered swarm adaptation",
    fontSize: 10,
    color: C.muted,
  });
}

export function label(slide, ctx, text, x, y, width, color = C.blue) {
  ctx.addText(slide, {
    x,
    y,
    width,
    height: 22,
    text,
    fontSize: 12,
    bold: true,
    color,
  });
}

export function metric(slide, ctx, x, y, width, value, labelText, note, color = C.blue) {
  ctx.addShape(slide, {
    x,
    y,
    width,
    height: 98,
    fill: C.paper,
    line: { style: "solid", fill: C.rule, width: 1 },
  });
  ctx.addText(slide, {
    x: x + 18,
    y: y + 12,
    width: width - 36,
    height: 34,
    text: value,
    fontSize: 28,
    bold: true,
    color,
    typeface: ctx.fonts.title,
  });
  ctx.addText(slide, {
    x: x + 18,
    y: y + 49,
    width: width - 36,
    height: 18,
    text: labelText,
    fontSize: 12,
    bold: true,
    color: C.ink,
  });
  if (note) {
    ctx.addText(slide, {
      x: x + 18,
      y: y + 70,
      width: width - 36,
      height: 15,
      text: note,
      fontSize: 9,
      color: C.muted,
    });
  }
}

export function bullet(slide, ctx, text, x, y, width, color = C.ink) {
  ctx.addShape(slide, {
    x,
    y: y + 8,
    width: 7,
    height: 7,
    fill: color,
    line: ctx.line(),
  });
  ctx.addText(slide, {
    x: x + 17,
    y,
    width,
    height: 48,
    text,
    fontSize: 15,
    color: C.ink,
  });
}

export function source(slide, ctx, text) {
  ctx.addText(slide, {
    x: 780,
    y: 681,
    width: 446,
    height: 18,
    text,
    fontSize: 9,
    color: C.muted,
    align: "right",
  });
}

export async function figure(slide, ctx, imagePath, x, y, width, height) {
  ctx.addShape(slide, {
    x,
    y,
    width,
    height,
    fill: C.paper,
    line: { style: "solid", fill: C.rule, width: 1 },
  });
  await ctx.addImage(slide, {
    path: imagePath,
    x: x + 10,
    y: y + 10,
    width: width - 20,
    height: height - 20,
    fit: "contain",
  });
}
