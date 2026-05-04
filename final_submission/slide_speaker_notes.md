# Premium Deck Presentation Script

This script matches `confidence_triggered_swarm_premium.pptx`, the current
10-slide deck for the May 4, 2026 presentation. It is written as a natural
rehearsal script, not something to read word-for-word. Aim for about 11 minutes.

## Delivery Style

- Sound like you are explaining the project to smart classmates, not reading a
  paper aloud.
- Use the numbers on the slides as anchors. Do not read every bullet.
- Keep the core claim honest: clean retention plus partial robustness under
  post-training surprise.
- Say clearly that the policy was trained clean only, then surprised after
  training.

## Slide 1 - Title, 0:00-0:55

"Hi everyone, this project is about confidence-triggered lifelong adaptation for
drone swarms under post-training surprises.

The main question is pretty practical: if we train a drone swarm policy in a
clean simulator, what happens when deployment is not clean anymore? In other
words, the policy learned formation flight, but then at test time it sees wind,
sensor noise, actuator weakness, or shifted waypoints.

The thesis is not that we solved drone robustness completely. The claim is more
specific: a clean-trained PPO policy can use confidence signals to notice when
deployment looks unfamiliar, adapt between episodes, and still keep its original
clean skill reasonably intact. So the project is about the whole loop: train,
surprise-test, adapt, and audit forgetting."

Transition:
"I will start with why this problem matters, then I will walk through the
environment, the adaptation method, and the main results."

## Slide 2 - Motivation, 0:55-1:55

"The motivation is the deployment gap. Reinforcement learning policies usually
look much better in the simulator than they do when the environment shifts.
That matters a lot for drones because small physical changes can break a policy:
wind changes the dynamics, noisy sensors change what the policy thinks is
happening, weak actuators change the effect of actions, and a shifted waypoint
changes the task context.

There are two obvious but imperfect responses. One is retraining, which is
expensive and not always possible after deployment. The other is online
adaptation, but unconstrained adaptation can overwrite the original skill. That
is the catastrophic forgetting problem.

So the approach here has three parts. First, detect low confidence using policy
entropy and Monte Carlo dropout variance. Second, adapt between episodes using
reward-weighted fine-tuning, but only from episodes that pass a quality gate.
Third, protect the clean skill using KL anchoring, clean replay, and EWC. The
goal is not to make the policy perfect under every surprise. The goal is to
recover some performance without destroying what it already knew."

Transition:
"Now here is the actual task and what the policy was trained on."

## Slide 3 - Task And Environment, 1:55-3:05

"The environment is `FormationAviary`, built on PyBullet. The task uses two CF2X
drones with a shared-policy IPPO setup. Each episode is 15 simulated seconds,
which is 450 control steps at 30 hertz. Each drone gets 75 observation
dimensions: the kinematic state plus the offset to the current waypoint.

This detail is important: the baseline was trained on clean episodes only. It
was not trained with the surprise conditions mixed in. The surprise wrapper is
applied after training, during evaluation and adaptation.

The surprise suite has four levels. Clean has no perturbation. Mild adds small
wind and sensor noise. Moderate increases wind and sensor noise and weakens the
actuator to 0.85. Severe pushes this further: max wind 0.10, sensor noise 0.05,
actuator strength 0.70, and occasional goal shifts.

The reward combines waypoint tracking, formation keeping, an alive bonus, and
boundary penalties. A useful way to think about it is that the drones need to
move the formation centroid through waypoints while keeping the two-drone
formation stable and avoiding bad states like low altitude or excessive tilt."

Transition:
"Given that setup, the next question is how the policy decides when to adapt."

## Slide 4 - Method, 3:05-4:30

"The adaptation loop has four steps.

First, the clean baseline policy is deployed in the surprised environment. At
each step, I compute confidence from two uncertainty signals. One is policy
entropy: if the action distribution is broad, the policy is less confident. The
second is MC-dropout variance: I run 10 stochastic forward passes and measure how
much the predicted action changes. Those signals are calibrated on clean
episodes, then tracked over a 30-reading window.

Second, the method checks whether the episode is useful for adaptation. This is
important because adapting from a crash can make the policy worse. The default
gate requires at least 30 steps and mean reward at least negative 5. If the
confidence is low but the episode is junk, the system skips adaptation.

Third, when the trigger fires, the method runs reward-weighted behavior cloning
on recent experience. Higher-reward transitions get more gradient weight. The
update is small: 5 epochs, learning rate 1e-4, and updates happen only between
episodes.

Fourth, the update is constrained. KL anchoring keeps the adapted policy close
to the frozen clean policy, clean replay mixes clean transitions into adaptation
batches, and EWC penalizes changes to parameters that mattered for the original
clean behavior."

Transition:
"So with that method, here is what happened compared with the frozen baseline."

## Slide 5 - Main Results, 4:30-5:45

"This is the main result, and I want to be careful about the wording. The method
shows partial robustness. It does not win everywhere.

On clean episodes, lifelong is slightly higher than frozen: 1358.9 versus
1305.2, which is plus 4.1 percent. On mild surprise, the improvement is much
larger: 159.7 versus 105.9, about plus 50.8 percent. On severe, it is also
higher: 42.2 versus 27.3, about plus 54.7 percent.

But moderate is the honest failure case. Frozen gets 49.4 and lifelong gets
45.2, which is minus 8.5 percent. So the correct interpretation is not
'adaptation always wins.' The correct interpretation is that confidence-triggered
adaptation can help under some post-training shifts while preserving clean
behavior, but the trigger and adaptation objective still need work.

Also, these canonical results are single-seed results with seed 42, so I would
treat them as directional rather than statistically final."

Transition:
"Because lifelong adaptation can easily cause forgetting, the next slide checks
whether the clean skill survived."

## Slide 6 - Forgetting Analysis, 5:45-6:45

"This slide is the clean-after-adaptation audit. The policy was adapted under
the hardest surprise condition, severe, and then evaluated again on clean
FormationAviary.

Before adaptation, the clean reward in this probe was 1308.9. After severe
adaptation, clean reward was 1386.1. The evaluator reports
`forgetting_detected: false`.

I would not overclaim this as the policy becoming universally better. The
waypoint count is slightly lower after adaptation, so the safest wording is
clean retention, not perfect improvement. But the important thing is that the
adaptation did not collapse the original clean behavior. That supports the value
of the anti-forgetting safeguards."

Transition:
"The single before-and-after test is useful, but we also wanted a more continual
learning style view."

## Slide 7 - Continual Learning, 6:45-8:00

"This matrix is the most important slide for the continual-learning part of the
project.

The rows show the phase the lifelong policy has just adapted through: after
clean, after mild, after moderate, and after severe. The columns show what it is
evaluated on afterward. The key column is the first one: clean.

If adaptation were causing catastrophic forgetting, we would expect the clean
column to collapse as the policy moves through mild, moderate, and severe. That
is not what happens. The clean values stay in the same range: 1316.0 after
clean, 1261.3 after mild, 1406.2 after moderate, and 1322.9 after severe.

The continual-learning metrics tell a mixed but useful story. Backward transfer
is plus 15.4, forward transfer is plus 17.2, and remembering is 1.0. At the
same time, final average reward is 393.1, below the frozen reference of 412.9.
So again, this is not a victory lap. It says clean retention looks good, but
overall adaptation performance is still uneven."

Transition:
"To understand why the method behaves this way, I also ran ablations on the
safeguards."

## Slide 8 - Ablation Study, 8:00-9:05

"The ablation study focuses on severe surprise because that is the harshest
setting. The full method has the lowest variance, which means it is the most
stable in this run, but it is not the highest mean reward.

When EWC or KL anchoring is removed, the mean severe reward can spike higher,
but the variance also increases and the clean-skill guardrail is weaker. That is
the tradeoff: if the goal is only short-run severe reward, removing safeguards
can look attractive. But if the goal is lifelong adaptation without forgetting,
those safeguards are part of the actual objective.

The main takeaway is that the bottleneck is probably not simply 'which
safeguard should we delete.' The bigger issue is trigger sensitivity and data
quality. The method adapts conservatively, and when episodes are short or low
quality, it avoids learning from them. That protects the policy, but it also
limits how much recovery is possible."

Transition:
"Before closing, I want to clarify the contribution split briefly."

## Slide 9 - Author Contributions, 9:05-9:45

"I will keep this brief. For the technical side, I built the implementation,
including the environment setup, surprise wrapper, PPO/IPPO training, confidence
monitor, adaptation loop, anti-forgetting safeguards, evaluation scripts,
ablations, continual-learning evaluation, figures, and the final deck.

Deveshi helped with project framing during the proposal stage and contributed
to proposal and final-report writing and review. Ron helped with proposal and
final-report writing and review, and with feedback on how to explain the method
and results clearly.

The reason I am being explicit here is that the course asks us to state who
worked on what, so this slide is meant to be factual rather than dramatic."

Transition:
"I will wrap up with what I think the project actually shows and what I would
do next."

## Slide 10 - Conclusion, 9:45-11:10

"The full project built a pipeline that goes from clean training to surprise
testing, confidence-triggered adaptation, forgetting checks, ablations, and a
continual-learning matrix.

The strongest supported conclusion is that a clean-trained drone swarm policy
can retain clean behavior while partially recovering from post-training
surprises. The clearest gains are mild and severe, at plus 50.8 percent and plus
54.7 percent. The forgetting probes are also encouraging: clean reward is
retained after severe adaptation, and the sequential matrix keeps clean
performance stable across later phases.

The limitations are just as important. This is one seed, two drones, simplified
PyBullet physics, and no real hardware. Moderate severity regresses in the main
evaluation. The adaptation rate is also low, around 2 percent in the main run,
which suggests the confidence trigger is conservative.

If I extended this, I would first run multiple seeds and larger swarms, because
that would tell us whether the result is robust. Then I would improve the
adaptation objective so the policy learns from surprise episodes more
effectively. Finally, I would look at peer-help mechanisms between drones, where
one drone's confidence or trajectory could help another adapt more safely.

So the final takeaway is: this is not a solved sim-to-real drone system, but it
is a concrete lifelong-learning pipeline showing how uncertainty-triggered
adaptation can recover under deployment shift while explicitly auditing
forgetting."

## If Running Long

- On slide 3, skip the reward details and only say clean training plus
  post-training surprise suite.
- On slide 7, do not read the CL metrics. Say the clean column stays stable and
  remembering is 1.0.
- On slide 8, use only this sentence: "The ablations show a stability versus
  peak-reward tradeoff, and the real bottleneck is trigger and data quality."

## Likely Q&A Answers

**Was it trained with surprises?**  
No. The baseline PPO/IPPO policy was trained on clean `FormationAviary` only.
Wind, noise, actuator weakness, dropout, and goal shifts were injected after
training during evaluation/adaptation.

**Why does moderate get worse while severe improves?**  
The adaptation trigger and episode-quality gate are conservative. Moderate can
sit in a region where the policy is degraded but does not always produce enough
useful adaptation data. Severe produces clearer low-confidence signals, but the
result is still noisy and single-seed.

**Why keep safeguards if ablations sometimes get higher reward?**  
Because the objective is not only severe reward. The objective is adaptation
without forgetting. Removing safeguards can raise short-run reward but weakens
the clean-skill protection that the project is testing.

**What is novel here?**  
The novelty is the integrated pipeline: post-training surprise benchmark,
dual-signal confidence trigger, between-episode adaptation with
anti-forgetting constraints, and explicit clean-after-surprise forgetting audits
for the drone swarm setting.
