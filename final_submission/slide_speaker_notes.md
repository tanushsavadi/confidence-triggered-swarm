# Premium Deck Presentation Script

This script matches `confidence_triggered_swarm_premium.pptx`, the current
10-slide deck for the May 4, 2026 presentation. It is written as a natural
rehearsal script, not something to read word for word. Aim for about 11 minutes.

## Delivery Style

- Do not read the slide text back to the audience.
- Use each slide as evidence, then explain what the evidence means.
- Sound like you are walking smart classmates through the project.
- Keep the core claim honest. This is clean retention plus partial robustness
  under post-training surprise.
- Say clearly that the policy was trained clean only, then surprised after
  training.

## Slide 1 - Title - about 55 seconds

"Hi everyone, this project is about confidence-triggered lifelong adaptation for
drone swarms under post-training surprises.

The easiest way to frame the project is this. Training a policy in simulation is
one thing. Trusting that same policy after the environment changes is a harder
problem. In this project, the policy learns clean two-drone formation flight
first, and then we ask what happens when the world becomes less clean after
training.

The thesis box gives the formal claim, but I want to phrase it more simply. I am
not claiming that this solves drone robustness. I am claiming that confidence
can be used as a trigger. When the policy starts acting uncertain, it can adapt
between episodes while still being constrained to remember the clean skill.

So the story of the talk is not just a method. It is the whole experimental
pipeline. Train on clean formation flight, inject surprise after training,
adapt only when confidence drops, and then check whether the original behavior
survived."

Transition
"I will start with why this kind of post-training surprise matters."

## Slide 2 - Motivation - about 60 seconds

"This slide is really about why clean simulator performance is not enough.

On the left, the slide lists the kinds of shifts that are realistic for drones.
I will not read every item, but the intuition is that each one changes a
different part of the control problem. Wind changes the physics. Sensor noise
changes perception. Actuator weakness changes whether an action has the effect
the policy expects. Goal shifts change the task context.

The reason this matters for lifelong learning is that both simple solutions are
unsatisfying. If we retrain from scratch every time, the system is expensive and
not very adaptive. If we allow unrestricted updates, the policy may adapt to the
new condition by forgetting the original clean formation skill.

That is why the right side of the slide is organized as detect, adapt, and
protect. The important part is the combination. Confidence decides when the
system should even consider adapting. The quality gate tries to avoid learning
from bad crash data. The anti-forgetting terms keep the update from drifting too
far away from the original policy."

Transition
"With that motivation, here is the concrete drone task used for the project."

## Slide 3 - Task And Environment - about 70 seconds

"This slide gives the experimental setup. I am not going to walk through every
number, but there are three details that matter for interpreting the results.

First, this is a small but real multi-agent control setting. There are two CF2X
drones using a shared-policy IPPO setup. The policy is not controlling a toy
point mass. It is acting through the PyBullet drone environment.

Second, the observation includes both drone state and waypoint information. That
means the policy has enough context to track the formation goal, but it still
has to handle noisy or shifted inputs once surprises are added.

Third, and this is the key design choice, the note at the bottom says the
baseline is trained on clean episodes only. The surprise suite is not part of
baseline training. As you can see in the table, mild, moderate, and severe
progressively add more shift after training. That lets us ask whether the
lifelong layer helps after the clean policy has already been learned.

The reward section is there to show what success means. The drones are rewarded
for moving through waypoints while keeping formation and avoiding unstable
states. So when reward collapses later, it is not just a random metric. It means
the controller is failing at the formation task."

Transition
"Now I will explain how the adaptation layer decides when and how to update."

## Slide 4 - Method - about 85 seconds

"This slide is a method diagram, but the key idea is simple. The policy does not
adapt constantly. It adapts only when its own confidence suggests that the
current environment is outside what it learned cleanly.

The first part of the loop measures uncertainty. Entropy tells us when the
policy distribution is broad. MC dropout gives a second signal by checking
whether the network predictions change across repeated stochastic forward
passes. Using both is useful because we do not want the trigger to depend on
only one noisy signal.

The second part is the safety check. This is important because a bad episode can
be worse than no data at all. If the drones crash immediately, that episode does
not teach a useful recovery behavior. It mostly teaches the policy to imitate a
failure. That is why the update is gated by episode length and mean reward.

The third part is the update itself. It is reward-weighted behavior cloning, so
the policy leans more toward transitions that worked better. This is not a full
retraining step. It is a small between-episode adjustment.

The last part is what makes it a lifelong-learning experiment rather than only a
robustness experiment. KL anchoring, clean replay, and EWC all push against
catastrophic forgetting in different ways. The point is to adapt without
letting the policy forget the clean formation behavior."

Transition
"Now we can look at whether that actually helped."

## Slide 5 - Main Results - about 75 seconds

"This is the main result slide. The first thing to notice in the graph is the
pattern, not just the individual numbers.

The clean condition stays strong, which is important because the method should
not damage the original task. Mild and severe show clear gains for the lifelong
policy over the frozen baseline. As shown by the cards, mild improves by about
50.8 percent and severe improves by about 54.7 percent.

The moderate condition is where the story becomes more honest and more
interesting. The lifelong policy is slightly worse there. That tells us the
method is not a universal robustness fix. It helps in some shifted conditions,
but the trigger and update objective are still imperfect.

So if I had to summarize this slide in one sentence, I would say the method
shows partial robustness while preserving clean behavior. The honest note is
also important. These are single-seed results with seed 42, so the right way to
read the graph is as directional evidence, not as a statistically final result."

Transition
"Since adaptation can help reward but still cause forgetting, the next slide
checks the clean skill directly."

## Slide 6 - Forgetting Analysis - about 60 seconds

"This slide answers a different question from the previous one. Slide 5 asks
whether reward improves under surprise. This slide asks whether adaptation
damages the original clean task.

As the figure shows, the clean reward after severe adaptation is still high. In
fact, it is higher in this probe than before adaptation. I would not present
that as proof that severe adaptation improves clean flight in general. The more
careful interpretation is that there is no catastrophic forgetting in this run.

The small caveat on the slide matters too. Waypoints are slightly lower after
adaptation, so this is not a perfect clean-skill improvement story. But the main
signal is that the policy did not collapse when evaluated again on clean
FormationAviary.

That is exactly why the anti-forgetting pieces are included. Without this kind
of clean re-test, it would be easy to report surprise recovery while missing the
fact that the original skill was damaged."

Transition
"A single clean re-test is useful, but the next slide gives a stronger continual
learning view."

## Slide 7 - Continual Learning - about 75 seconds

"This matrix is the strongest evidence for the forgetting story.

The way to read it is not to scan every cell. Focus on the clean column. Each
row is after the policy has gone through another adaptation phase. If the
lifelong process were forgetting the clean task, the clean column would drop as
we move downward through the phases.

As shown in the table, that collapse does not happen. The clean reward stays in
the same general range after mild, after moderate, and after severe adaptation.
That is the main reason this slide matters.

The metrics on the right give a compact summary. Backward transfer and forward
transfer are positive, and remembering is 1.0. At the same time, the final
average reward is lower than the frozen reference. So the slide supports a
careful conclusion. The method is good at retaining clean behavior in this run,
but it is not yet better than frozen performance on every aggregate metric."

Transition
"The next question is what the safeguards are doing and whether they are worth
keeping."

## Slide 8 - Ablation Study - about 65 seconds

"This slide is useful because it prevents an overly simple conclusion.

If we only wanted the highest severe reward in a short run, the ablations might
tempt us to remove safeguards. As shown in the figure, some removed-safeguard
variants can reach a higher mean reward than the full method.

But the full method has the lowest variance, which means it is the most stable
in this severe-surprise ablation. That matters because the project is not only
about maximizing one severe reward number. It is about adapting while protecting
the clean policy.

So the real takeaway is the one at the bottom of the slide. The safeguards are
not the only bottleneck. The harder problem is deciding when adaptation data is
good enough to learn from. The trigger is conservative, and the quality gate
protects the policy, but it also limits how much recovery can happen."

Transition
"Before the final takeaway, I want to briefly cover the contribution slide."

## Slide 9 - Author Contributions - about 40 seconds

"I will keep this brief because the slide already lists the split.

The main thing to say is that the technical implementation and experimental
pipeline were primarily my work. That includes the environment setup, surprise
wrapper, PPO and IPPO training, confidence monitor, adaptation loop, safeguards,
evaluation scripts, ablations, continual-learning evaluation, figures, and deck.

Deveshi helped during the proposal and report stages, especially with framing,
writing, review, and presentation organization. Ron also helped with proposal
and report writing and with feedback on how to explain the method and results
clearly.

I am including this because the course specifically asks us to communicate who
worked on what. The goal here is to be clear and factual."

Transition
"I will close by separating what the project shows from what it does not show."

## Slide 10 - Conclusion - about 85 seconds

"This final slide is the summary, but I want to use it to separate the strong
claim from the limitations.

The strong claim is that the full pipeline works as an experimental framework.
It trains a clean policy, exposes it to post-training surprises, adapts based on
confidence, and then audits whether clean behavior survived. As shown in the
achieved section, the clearest robustness gains are mild and severe, and the
clean-skill checks are encouraging.

The limitation section is just as important. This is not enough evidence to
claim a solved drone deployment system. It is one seed, two drones, simplified
simulation physics, and no hardware. Moderate surprise also regresses, which
means the adaptation trigger and objective need more work.

The future-work list points to the next logical steps. Multi-seed validation
would test whether the pattern is reliable. Larger swarms would test whether
the idea scales beyond two drones. Better adaptation objectives could make the
updates more useful instead of only more cautious. Peer-help mechanisms would
also be interesting because one drone's confidence signal could help another
drone adapt more safely.

So the final takeaway is this. The project does not solve sim-to-real drone
control, but it gives a concrete lifelong-learning pipeline for post-training
surprise. The most defensible result is clean retention plus partial robustness,
with clear next steps for making the adaptation stronger."

## If Running Long

- On slide 3, skip the reward details and only say clean training plus
  post-training surprise suite.
- On slide 7, do not read the continual-learning metrics. Say the clean column
  stays stable and remembering is 1.0.
- On slide 8, use only this sentence. The ablations show a stability versus
  peak-reward tradeoff, and the real bottleneck is trigger and data quality.

## Likely Q&A Answers

**Was it trained with surprises?**  
No. The baseline PPO and IPPO policy was trained on clean `FormationAviary`
only. Wind, noise, actuator weakness, dropout, and goal shifts were injected
after training during evaluation and adaptation.

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
