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
"I will start with why I framed the project around post-training surprise."

## Slide 2 - Motivation - about 50 seconds

"Clean simulator performance is only the starting point. For drones, the real
question is what happens after training when the environment stops matching the
clean simulator.

When I built the surprise setup, I wanted it to cover changes a policy could
actually face after deployment. Wind changes the dynamics. Sensor noise changes
what the policy observes. Actuator weakness changes how much control the drone
really has. Goal shifts test whether the policy can handle a changed mission
context.

That is why I built the method around three steps. Detect when confidence drops,
adapt only from usable experience, and protect the clean policy while updating.
Retraining after every shift is expensive, and updating freely can erase the
formation skill we trained for.

So the point of this project is not just to make the surprise environment harder.
The point is to test whether a clean-trained policy can notice that shift, adapt
to it, and still remember the original task."

Transition
"That leads into the drone task I used to test the idea."

## Slide 3 - Task And Environment - about 60 seconds

"For the environment, I kept the setup small enough to run end to end, but still
grounded in drone control.

I used two CF2X drones in PyBullet with a shared IPPO policy. Each episode gives
the policy 450 control steps, and each drone observes both its state and
waypoint information. So the policy has enough context to fly the route, but the
surprises can still break the behavior after training.

The most important design choice is that baseline training stays clean. I did
not train the baseline with wind, sensor noise, actuator weakness, or goal
shifts mixed in. Those are added only after training, so the evaluation is really
testing how the learned clean controller handles deployment shift.

As you can see in the table, the severities add more shift as we move from mild
to moderate to severe. The reward then checks whether the drones still move
through waypoints while keeping formation and avoiding unstable states."

Transition
"Next, I will show how I made the policy decide when to adapt."

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

"The main results have a clear pattern. The method helps in some conditions, but
not all of them.

Clean stays strong, which matters because adaptation should not damage the
original task. Mild and severe show the clearest gains for the lifelong policy.
As shown by the cards, mild improves by about 50.8 percent and severe improves
by about 54.7 percent.

Moderate is the honest failure case. The lifelong policy is slightly worse
there, which tells us this is not a universal robustness fix. It helps in some
shifted conditions, but the trigger and update objective still need work.

What we learned is that our method gives partial robustness while still
preserving the clean behavior we trained for. These are single-seed results with
seed 42, so the graph is directional evidence, not a statistically final
conclusion."

Transition
"Since adaptation can help reward but still cause forgetting, the next slide
checks the clean skill directly."

## Slide 6 - Forgetting Analysis - about 50 seconds

"The next question is whether adaptation helped by sacrificing the original
skill. After adapting under severe surprise, does the policy still work on the
original clean task?

As the figure shows, clean reward after severe adaptation is still high. In this
probe, it is even higher than before adaptation. That does not mean severe
adaptation generally improves clean flight. The safer interpretation is that
there is no catastrophic forgetting in this run.

There is one caveat. Waypoints are slightly lower after adaptation, so this is
not a perfect improvement story. But the main signal is that the clean behavior
did not collapse.

I included this clean re-test because surprise recovery by itself would not be
enough. If the original formation skill was damaged, then the adaptation would
not really be lifelong learning."

Transition
"The next slide gives a stronger continual-learning view of that same issue."

## Slide 7 - Continual Learning - about 65 seconds

"The matrix gives a stronger version of the forgetting check.

The clean column is the key part of this matrix. Each row is after another
adaptation phase. If the policy were forgetting clean flight, that clean column
would drop as we move downward.

As shown in the table, that collapse does not happen. Clean reward stays in the
same general range after mild, after moderate, and after severe adaptation.
That is the main reason this slide matters.

The metrics give a compact summary. Backward transfer and forward transfer are
positive, and remembering is 1.0. But final average reward is still lower than
the frozen reference. So what I take from this is clean retention looks good,
while overall adaptation performance still needs work."

Transition
"Next, I will show what the safeguards are doing."

## Slide 8 - Ablation Study - about 55 seconds

"I included this ablation because it shows the tradeoff in the method.

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
"Before the final takeaway, here is the contribution split."

## Slide 9 - Author Contributions - about 30 seconds

"For contributions, the technical implementation and experimental pipeline were
primarily my work, including the environment, surprise wrapper, PPO and IPPO
training, confidence monitor, adaptation loop, safeguards, evaluations,
ablations, continual-learning results, figures, and deck. Deveshi and Ron
helped during the proposal and report stages with framing, writing, review, and
feedback on how to explain the method and results clearly."

Transition
"I will close with what the results support and what I would build next."

## Slide 10 - Conclusion - about 70 seconds

"To wrap up, the main result is clean retention with partial robustness, and the
next step is making that adaptation more reliable.

What we built is the full pipeline. We trained a clean drone policy, added
surprises after training, used confidence to decide when to adapt, and then
checked whether the clean behavior was still there.

What we learned is that confidence-triggered adaptation can preserve clean
behavior while recovering some performance under surprise. Mild and severe
improved, and the clean-skill checks were encouraging.

We also learned where the method needs work. Moderate got worse, and the
evidence is still scoped to one seed, two drones, simplified PyBullet physics,
and no hardware. So the next version should focus on a stronger trigger and a
better adaptation objective.

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
