# Premium Deck Presentation Script

This script matches `confidence_triggered_swarm_premium.pptx`, the current
10-slide deck for the May 4, 2026 presentation. It is written as a natural
rehearsal script, not something to read word for word. Aim for about 11 minutes.

## Delivery Style

- Sound like you are explaining the project to smart classmates, not reading a
  paper aloud.
- Use the numbers on the slides as anchors. Do not read every bullet.
- Keep the core claim honest. This is clean retention plus partial robustness
  under post-training surprise.
- Say clearly that the policy was trained clean only, then surprised after
  training.

## Slide 1 - Title - about 55 seconds

"Hi everyone, this project is about confidence-triggered lifelong adaptation for
drone swarms under post-training surprises.

The main question is pretty practical. If we train a drone swarm policy in a
clean simulator, what happens when deployment is not clean anymore? In other
words, the policy learned formation flight, but then at test time it sees wind,
sensor noise, actuator weakness, or shifted waypoints.

As shown in the thesis box at the bottom, the claim is not that we solved drone
robustness completely. The claim is more specific. A clean-trained PPO policy
can use confidence signals to notice when deployment looks unfamiliar, adapt
between episodes, and still keep its original clean skill reasonably intact.
So the project is about the whole loop. Train, surprise-test, adapt, and audit
forgetting."

Transition
"I will start with why this problem matters, then I will walk through the
environment, the adaptation method, and the main results."

## Slide 2 - Motivation - about 60 seconds

"The motivation is the deployment gap. Reinforcement learning policies usually
look much better in the simulator than they do when the environment shifts.
As you can see on the left side of the slide, the kinds of shifts we are talking
about are very natural for drones. Wind changes the dynamics. Noisy sensors
change what the policy thinks is happening. Weak actuators change the effect of
actions. Shifted waypoints change the mission context.

There are two obvious but imperfect responses. One is retraining, which is
expensive and not always possible after deployment. The other is online
adaptation, but unconstrained adaptation can overwrite the original skill. That
is the catastrophic forgetting problem.

The right side of the slide shows the three-part approach. First, detect low
confidence using policy entropy and Monte Carlo dropout variance. Second, adapt
between episodes using reward-weighted fine-tuning, but only from episodes that
pass a quality gate. Third, protect the clean skill using KL anchoring, clean
replay, and EWC. The goal is not to make the policy perfect under every
surprise. The goal is to recover some performance without destroying what it
already knew."

Transition
"Now here is the actual task and what the policy was trained on."

## Slide 3 - Task And Environment - about 70 seconds

"The environment is `FormationAviary`, built on PyBullet. As shown by the three
summary numbers at the top, the task uses two CF2X drones, 450 control steps per
episode, and 75 observation dimensions per drone. Each episode is 15 simulated
seconds, and each drone observes its kinematic state plus the offset to the
current waypoint.

The most important detail is in the note at the bottom. The baseline was trained
on clean episodes only. It was not trained with the surprise conditions mixed
in. The surprise wrapper is applied after training, during evaluation and
adaptation.

As you can see in the surprise table, the setup moves from clean to mild,
moderate, and severe. Clean has no perturbation. Mild adds small wind and sensor
noise. Moderate increases wind and sensor noise and weakens the actuator to
0.85. Severe pushes this further with max wind 0.10, sensor noise 0.05,
actuator strength 0.70, and occasional goal shifts.

The reward components are summarized on the right. The drones need to move the
formation centroid through waypoints while keeping the two-drone formation
stable and avoiding bad states like low altitude or excessive tilt."

Transition
"Given that setup, the next question is how the policy decides when to adapt."

## Slide 4 - Method - about 85 seconds

"This slide shows the adaptation loop. I will walk through it in order.

In the first panel, the clean baseline policy is deployed in the surprised
environment. At each step, I compute confidence from two uncertainty signals.
One is policy entropy. If the action distribution is broad, the policy is less
confident. The second is MC-dropout variance. I run 10 stochastic forward passes
and measure how much the predicted action changes. Those signals are calibrated
on clean episodes, then tracked over a 30-reading window.

In the second panel, the method checks whether the episode is useful for
adaptation. This matters because adapting from a crash can make the policy
worse. The default gate requires at least 30 steps and mean reward at least
negative 5. If confidence is low but the episode is junk, the system skips
adaptation.

In the third panel, when the trigger fires, the method runs reward-weighted
behavior cloning on recent experience. Higher-reward transitions get more
gradient weight. The update is intentionally small. It uses 5 epochs, learning
rate 1e-4, and it happens only between episodes.

The fourth panel is the anti-forgetting part. KL anchoring keeps the adapted
policy close to the frozen clean policy. Clean replay mixes clean transitions
into adaptation batches. EWC penalizes changes to parameters that mattered for
the original clean behavior."

Transition
"So with that method, here is what happened compared with the frozen baseline."

## Slide 5 - Main Results - about 75 seconds

"This is the main result, and I want to be careful about the wording. The method
shows partial robustness. It does not win everywhere.

As you can see in the graph and the four result cards, clean performance is
slightly higher with lifelong adaptation. It goes from 1305.2 to 1358.9, which
is plus 4.1 percent. On mild surprise, the improvement is much larger. It goes
from 105.9 to 159.7, which is about plus 50.8 percent. On severe, it also
improves from 27.3 to 42.2, which is about plus 54.7 percent.

But the moderate card is the honest failure case. Frozen gets 49.4 and lifelong
gets 45.2, which is minus 8.5 percent. So the correct interpretation is not
that adaptation always wins. The correct interpretation is that
confidence-triggered adaptation can help under some post-training shifts while
preserving clean behavior, but the trigger and adaptation objective still need
work.

The honest note on the slide matters too. These canonical results are
single-seed results with seed 42, so I would treat them as directional rather
than statistically final."

Transition
"Because lifelong adaptation can easily cause forgetting, the next slide checks
whether the clean skill survived."

## Slide 6 - Forgetting Analysis - about 60 seconds

"This slide is the clean-after-adaptation audit. The policy was adapted under
the hardest surprise condition, severe, and then evaluated again on clean
FormationAviary.

As the figure shows, before adaptation the clean reward in this probe was
1308.9. After severe adaptation, clean reward was 1386.1. The evaluator reports
that forgetting detected is false.

I would not overclaim this as the policy becoming universally better. The slide
also notes that the waypoint count is slightly lower after adaptation, so the
safest wording is clean retention, not perfect improvement. But the important
thing is that adaptation did not collapse the original clean behavior. That
supports the value of the anti-forgetting safeguards."

Transition
"The single before-and-after test is useful, but we also wanted a more continual
learning style view."

## Slide 7 - Continual Learning - about 75 seconds

"This matrix is the most important slide for the continual-learning part of the
project.

As shown by the table, the rows are the phase the lifelong policy has just
adapted through. After clean, after mild, after moderate, and after severe. The
columns show what it is evaluated on afterward. The key column is the first one,
which is clean.

If adaptation were causing catastrophic forgetting, we would expect the clean
column to collapse as the policy moves through mild, moderate, and severe. As
you can see, that is not what happens. The clean values stay in the same range.
They are 1316.0 after clean, 1261.3 after mild, 1406.2 after moderate, and
1322.9 after severe.

The metrics on the right tell a mixed but useful story. Backward transfer is
plus 15.4, forward transfer is plus 17.2, and remembering is 1.0. At the same
time, final average reward is 393.1, below the frozen reference of 412.9. So
again, this is not a victory lap. It says clean retention looks good, but
overall adaptation performance is still uneven."

Transition
"To understand why the method behaves this way, I also ran ablations on the
safeguards."

## Slide 8 - Ablation Study - about 65 seconds

"The ablation study focuses on severe surprise because that is the harshest
setting. As you can see in the ablation figure, the full method has the lowest
variance. That means it is the most stable in this run, but it is not the
highest mean reward.

When EWC or KL anchoring is removed, the mean severe reward can spike higher,
but the variance also increases and the clean-skill guardrail is weaker. That is
the tradeoff. If the goal is only short-run severe reward, removing safeguards
can look attractive. But if the goal is lifelong adaptation without forgetting,
those safeguards are part of the actual objective.

The takeaway at the bottom is the key point. The bottleneck is probably not
simply which safeguard should be deleted. The bigger issue is trigger
sensitivity and data quality. The method adapts conservatively, and when
episodes are short or low quality, it avoids learning from them. That protects
the policy, but it also limits how much recovery is possible."

Transition
"Before closing, I want to clarify the contribution split briefly."

## Slide 9 - Author Contributions - about 40 seconds

"I will keep this brief. As shown on this slide, for the technical side I built
the implementation, including the environment setup, surprise wrapper,
PPO/IPPO training, confidence monitor, adaptation loop, anti-forgetting
safeguards, evaluation scripts, ablations, continual-learning evaluation,
figures, and the final deck.

Deveshi helped with project framing during the proposal stage and contributed
to proposal and final-report writing and review. Ron helped with proposal and
final-report writing and review, and with feedback on how to explain the method
and results clearly.

The reason I am being explicit here is that the course asks us to state who
worked on what, so this slide is meant to be factual rather than dramatic."

Transition
"I will wrap up with what I think the project actually shows and what I would
do next."

## Slide 10 - Conclusion - about 85 seconds

"The final slide summarizes the full project. We built a pipeline that goes from
clean training to surprise testing, confidence-triggered adaptation, forgetting
checks, ablations, and a continual-learning matrix.

As shown in the achieved section, the strongest supported conclusion is that a
clean-trained drone swarm policy can retain clean behavior while partially
recovering from post-training surprises. The clearest gains are mild and severe,
at plus 50.8 percent and plus 54.7 percent. The forgetting probes are also
encouraging. Clean reward is retained after severe adaptation, and the
sequential matrix keeps clean performance stable across later phases.

The limitations section is just as important. This is one seed, two drones,
simplified PyBullet physics, and no real hardware. Moderate severity regresses
in the main evaluation. The adaptation rate is also low, around 2 percent in
the main run, which suggests the confidence trigger is conservative.

For future work, I would first run multiple seeds and larger swarms, because
that would tell us whether the result is robust. Then I would improve the
adaptation objective so the policy learns from surprise episodes more
effectively. Finally, I would look at peer-help mechanisms between drones, where
one drone's confidence or trajectory could help another adapt more safely.

So the final takeaway is this. This is not a solved sim-to-real drone system,
but it is a concrete lifelong-learning pipeline showing how
uncertainty-triggered adaptation can recover under deployment shift while
explicitly auditing forgetting."

## If Running Long

- On slide 3, skip the reward details and only say clean training plus
  post-training surprise suite.
- On slide 7, do not read the continual-learning metrics. Say the clean column
  stays stable and remembering is 1.0.
- On slide 8, use only this sentence. The ablations show a stability versus
  peak-reward tradeoff, and the real bottleneck is trigger and data quality.

## Likely Q&A Answers

**Was it trained with surprises?**  
No. The baseline PPO/IPPO policy was trained on clean `FormationAviary` only.
Wind, noise, actuator weakness, dropout, and goal shifts were injected after
training during evaluation and adaptation.

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
The novelty is the integrated pipeline. A post-training surprise benchmark,
dual-signal confidence trigger, between-episode adaptation with anti-forgetting
constraints, and explicit clean-after-surprise forgetting audits for the drone
swarm setting.
