# Premium Deck Presentation Script

This script matches `confidence_triggered_swarm_premium.pptx`, the current
10-slide deck for the May 4, 2026 presentation. It is written as a natural
rehearsal script, not something to read word for word. Aim for about 11 minutes.

## Delivery Style

- Do not read the slide text back to the audience.
- Use each slide as evidence, then explain what the evidence means.
- Keep a calm pace. It is better to say less clearly than to rush.
- Keep the core claim honest. The result is clean retention plus partial
  robustness under post-training surprise.
- Say clearly that the policy was trained clean only, then surprised after
  training.

## Slide 1 - Title - about 45 seconds

"Hi everyone, this project is about confidence-triggered lifelong adaptation for
drone swarms under post-training surprises.

The simple version is this. A drone policy can look good in a clean simulator,
but deployment is rarely that clean. In this project, the policy first learns
clean two-drone formation flight. Then after training, we expose it to surprises
like wind, sensor noise, actuator weakness, and shifted waypoints.

The thesis box gives the formal claim, but the main idea is that confidence can
act as a trigger. When the policy seems uncertain, it adapts between episodes,
while the safeguards try to preserve the clean skill.

So the project is really about the full loop. Train clean, add surprise after
training, adapt only when confidence drops, and then check whether the original
behavior survived."

Transition
"I will start with why this post-training surprise setting matters."

## Slide 2 - Motivation - about 50 seconds

"Clean simulator performance is a useful starting point, but for drones it is
not the full story.

In the Problem section, the examples are all realistic ways deployment can
shift. Wind changes the physics. Sensor noise changes what the policy thinks it
sees. Actuator weakness changes whether actions work as expected. Goal shifts
change the task context.

The issue is that the two simple answers are not enough. Retraining every time
is expensive. Updating freely can cause the policy to forget the clean formation
skill.

In Our Approach, the three pieces are detect low confidence, adapt only from
usable experience, and protect the original clean policy. The important part is
the combination. Confidence starts the update, the quality gate filters bad
episodes, and the anti-forgetting terms keep the update from drifting too far."

Transition
"With that motivation, here is the concrete drone task."

## Slide 3 - Task And Environment - about 60 seconds

"For the environment, the details I care about are the ones that affect how we
read the results.

The task uses two CF2X drones with a shared-policy IPPO setup in PyBullet. Each
episode has 450 control steps, and each drone observes its state plus waypoint
information. So it is a small setup, but it is still a real control problem,
not a toy point-mass task.

The key design choice is in the note at the bottom. The baseline is trained on
clean episodes only. The surprise suite is added after training.

As you can see in the table, the severities add more shift as we move from mild
to moderate to severe. That is what lets us test whether the lifelong layer can
recover after the clean policy has already been learned.

The reward section tells us what success means. The drones need to move through
waypoints while keeping formation and avoiding unstable states."

Transition
"Now I will explain how the adaptation layer decides when to update."

## Slide 4 - Method - about 70 seconds

"This method diagram has four pieces, but the core idea is simple. The policy
does not adapt all the time. It adapts only when confidence suggests that the
environment is outside the clean training distribution.

The confidence signal combines entropy and MC dropout. Entropy tells us when the
action distribution is broad. MC dropout checks whether repeated stochastic
forward passes disagree. Using both makes the trigger less dependent on one
noisy signal.

The quality gate is just as important. If the drones crash right away, that is
not useful data. Learning from that can make the policy imitate failure. So the
episode needs to be long enough and have enough reward before it enters the
adaptation buffer.

The update itself is small and happens between episodes. It uses
reward-weighted behavior cloning, so better transitions matter more. Then KL
anchoring, clean replay, and EWC push against forgetting. That is what makes the
method a lifelong-learning experiment instead of only a robustness test."

Transition
"Now we can look at whether that actually helped."

## Slide 5 - Main Results - about 65 seconds

"For the main results, I would read the graph by looking at the pattern first,
not by treating every number as equally important.

Clean stays strong, which matters because adaptation should not damage the
original task. Mild and severe show the clearest gains for the lifelong policy.
As shown by the cards, mild improves by about 50.8 percent and severe improves
by about 54.7 percent.

Moderate is the honest failure case. The lifelong policy is slightly worse
there, which tells us this is not a universal robustness fix. It helps in some
shifted conditions, but the trigger and update objective still need work.

So the careful one-sentence takeaway is partial robustness while preserving
clean behavior. Also, these are single-seed results with seed 42, so I would
treat the graph as directional evidence, not a statistically final conclusion."

Transition
"Since adaptation can help reward but still cause forgetting, the next slide
checks the clean skill directly."

## Slide 6 - Forgetting Analysis - about 50 seconds

"The next question is whether adaptation helped by sacrificing the original
skill. After adapting under severe surprise, does the policy still work on the
original clean task?

As the figure shows, clean reward after severe adaptation is still high. In this
probe, it is even higher than before adaptation. I would not overclaim that as a
general clean improvement. The safer interpretation is that there is no
catastrophic forgetting in this run.

There is one caveat. Waypoints are slightly lower after adaptation, so this is
not a perfect improvement story. But the main signal is that the clean behavior
did not collapse.

That clean re-test is important because surprise recovery alone would not be
enough if the original formation skill was damaged."

Transition
"The next slide gives a stronger continual-learning view of that same issue."

## Slide 7 - Continual Learning - about 65 seconds

"The matrix gives a stronger version of the forgetting check.

I would read it by following the clean column. Each row is after another
adaptation phase. If the policy were forgetting clean flight, that clean column
would drop as we move downward.

As shown in the table, that collapse does not happen. Clean reward stays in the
same general range after mild, after moderate, and after severe adaptation.
That is the main reason this slide matters.

The metrics give a compact summary. Backward transfer and forward transfer are
positive, and remembering is 1.0. But final average reward is still lower than
the frozen reference. So the conclusion is careful. Clean retention looks good,
but the method is not better than frozen on every aggregate metric."

Transition
"Next, I will show what the safeguards are doing."

## Slide 8 - Ablation Study - about 55 seconds

"The ablation result is useful because it keeps the conclusion honest.

If we only wanted the highest severe reward in a short run, some ablations might
look tempting. As shown in the figure, removing safeguards can raise the mean
reward.

But the full method has the lowest variance, which means it is the most stable
in this severe-surprise ablation. That matters because the goal is not only
short-run reward. The goal is adaptation while protecting the clean policy.

So the real bottleneck is not just which safeguard to delete. The harder problem
is deciding when adaptation data is good enough to learn from. The trigger is
conservative, and the quality gate protects the policy, but it also limits how
much recovery can happen."

Transition
"Before the final takeaway, I want to briefly cover the contribution slide."

## Slide 9 - Author Contributions - about 35 seconds

"For contributions, the course asks us to be clear about who worked on what, so
I want to state that directly.

The technical implementation and experimental pipeline were primarily my work.
That includes the environment, surprise wrapper, PPO and IPPO training,
confidence monitor, adaptation loop, safeguards, evaluations, ablations,
continual-learning results, figures, and deck.

Deveshi helped during the proposal and report stages with framing, writing,
review, and presentation organization. Ron also helped with proposal and report
writing, plus feedback on explaining the method and results.

I am including this because the course specifically asks us to communicate who
worked on what."

Transition
"I will close by separating what the project shows from what it does not show."

## Slide 10 - Conclusion - about 70 seconds

"To wrap up, I want to separate the strong claim from the limitations.

The strong claim is that the pipeline works as an experimental framework. It
trains a clean policy, exposes it to post-training surprises, adapts based on
confidence, and audits whether clean behavior survived. As shown in the
achieved section, the clearest robustness gains are mild and severe, and the
clean-skill checks are encouraging.

The limitations are just as important. We only tested one seed, two drones,
simplified simulation physics, and no hardware. Moderate surprise regresses,
which means the adaptation trigger and objective need more work.

The next steps are pretty clear. Multi-seed validation would test reliability.
Larger swarms would test scaling. Better adaptation objectives could make the
updates more useful instead of only cautious. Peer-help mechanisms are also
interesting because one drone's confidence could help another adapt more
safely.

So the final takeaway is this. The project does not solve sim-to-real drone
control, but it gives a concrete lifelong-learning pipeline for post-training
surprise. The most defensible result is clean retention plus partial
robustness."

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
