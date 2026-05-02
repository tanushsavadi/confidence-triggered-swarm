import { base, bullet, C, label, source } from "./common.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Adaptation loop", "The adaptation layer is conservative by design.", "It adapts only between episodes and uses three safeguards against catastrophic forgetting.");

  const lanes = [
    ["Confidence trigger", "Entropy and MC-dropout variance are calibrated on clean episodes.", C.blue],
    ["Episode quality gate", "Short or very low-reward episodes are rejected before replay.", C.orange],
    ["Reward-weighted update", "Usable transitions are weighted toward better-rewarded behavior.", C.green],
    ["Anti-forgetting", "KL anchor, clean replay, and EWC constrain drift from clean skill.", C.red],
  ];
  lanes.forEach((lane, i) => {
    const y = 196 + i * 104;
    ctx.addShape(slide, { x: 84, y, width: 260, height: 76, fill: lane[2], line: ctx.line() });
    ctx.addText(slide, { x: 108, y: y + 18, width: 214, height: 42, text: lane[0], fontSize: 18, bold: true, color: "#FFFFFF" });
    ctx.addShape(slide, { x: 390, y, width: 734, height: 76, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
    ctx.addText(slide, { x: 420, y: y + 18, width: 650, height: 40, text: lane[1], fontSize: 18, color: C.ink });
    if (i < lanes.length - 1) {
      ctx.addShape(slide, { x: 210, y: y + 80, width: 8, height: 24, fill: C.rule, line: ctx.line() });
    }
  });

  label(slide, ctx, "Default adaptation values", 84, 600, 280);
  bullet(slide, ctx, "threshold 0.5, window 30, 5 adaptation epochs, LR 1e-4, EWC lambda 1000, clean replay ratio 0.2, KL anchor 0.5.", 88, 624, 1030, C.blue);
  source(slide, ctx, "Source: default.yaml, confidence.py, lifelong_trainer.py");
  return slide;
}
