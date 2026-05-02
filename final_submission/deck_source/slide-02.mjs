import { base, bullet, C, label, metric, source } from "./common.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Project arc", "The work built a full clean-train, surprise-test, adapt-and-audit loop.", "This slide is the whole process in one view: training code, surprise wrapper, confidence monitor, lifelong updates, and evaluation artifacts.");

  const y = 238;
  const stages = [
    ["1", "Train clean IPPO", "FormationAviary only; no wind/noise/failure"],
    ["2", "Deploy to surprises", "Wind, sensor noise, actuator weakness, goal shift"],
    ["3", "Trigger on confidence", "Policy entropy plus MC-dropout variance"],
    ["4", "Audit forgetting", "Clean replay, EWC, KL, ablations, continual matrix"],
  ];
  stages.forEach((s, i) => {
    const x = 72 + i * 292;
    ctx.addShape(slide, { x, y, width: 232, height: 148, fill: C.paper, line: { style: "solid", fill: C.rule, width: 1 } });
    ctx.addText(slide, { x: x + 18, y: y + 16, width: 32, height: 30, text: s[0], fontSize: 22, bold: true, color: C.orange });
    ctx.addText(slide, { x: x + 58, y: y + 16, width: 154, height: 42, text: s[1], fontSize: 16, bold: true, color: C.ink });
    ctx.addText(slide, { x: x + 18, y: y + 64, width: 190, height: 56, text: s[2], fontSize: 13, color: C.muted });
    if (i < stages.length - 1) {
      ctx.addShape(slide, { x: x + 246, y: y + 70, width: 28, height: 4, fill: C.orange, line: ctx.line() });
    }
  });

  label(slide, ctx, "Environment values", 72, 440, 240);
  bullet(slide, ctx, "2 CF2X drones, PyBullet `PYB`, velocity control, kinematic observations plus waypoint offset.", 76, 474, 500);
  bullet(slide, ctx, "240 Hz physics, 30 Hz control, 15 second episodes, 0.5 m target formation spacing.", 76, 522, 500);
  bullet(slide, ctx, "Default waypoints: [0.15, 0.15, 0.5], [0.4, 0.4, 0.5], [0.6, 0.0, 0.7].", 76, 570, 500);

  label(slide, ctx, "Training values", 705, 440, 240, C.green);
  metric(slide, ctx, 708, 474, 168, "1e6", "timesteps", "seed 42", C.green);
  metric(slide, ctx, 900, 474, 168, "1e-4", "PPO LR", "gamma 0.99", C.blue);
  metric(slide, ctx, 1078, 474, 140, "2x256", "MLP", "tanh", C.orange);
  source(slide, ctx, "Source: confidence_triggered_swarm/configs/default.yaml");
  return slide;
}
