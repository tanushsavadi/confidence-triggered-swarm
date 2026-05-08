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

## Slide 4 - Method - about 85 seconds

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
method a lifelong-learning experiment instead of only a robustness test.

The prior work I am using here is continual learning and catastrophic forgetting.
Van de Ven, Soures, and Kudithipudi describe continual learning as learning from
changing data without overwriting what was learned before.

That is the reason I used EWC, clean replay, and KL anchoring. Those ideas come
from the broader continual learning problem. My novelty is not claiming that I
invented those safeguards. My contribution is applying them in this post-training
drone surprise setting, with confidence deciding when adaptation should happen."

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
preserving the clean behavior we trained for. This slide is a seed-42 diagnostic
from the presentation deck; the final written report adds a three-seed
validation table and keeps the same honest conclusion: partial recovery, not a
universal win."

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

The metrics give a compact summary. Backward transfer and forward transfer are
positive, and remembering is 1.0. But final average reward is still lower than
the frozen reference. So what I take from this is clean retention looks good,
while overall adaptation performance still needs work.

In my results, the safeguards agree with prior work on retention, but I still
cannot claim the adaptation problem is solved. The final average being lower
than frozen shows that this balance still needs work."

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

What we learned matches the continual learning framing. The anti-forgetting
pieces helped preserve clean behavior, which agrees with prior work on replay
and regularization. But the moderate regression shows the other side of the
tradeoff. Protecting the old skill is not enough if the adaptation trigger and
update objective are not reliable.

So my conclusion is that our method shows clean retention and partial
robustness, but it does not solve lifelong drone adaptation yet.

The next steps are pretty clear. Multi-seed validation would test reliability.
Larger swarms would test scaling. Better adaptation objectives could make the
updates more useful instead of only cautious. Peer-help mechanisms are also
interesting because one drone's confidence could help another adapt more
safely.

That is the claim I would defend. The project does not solve sim-to-real drone
control, but it gives a concrete lifelong-learning pipeline for post-training
surprise. The result I can defend is clean retention plus partial robustness."

## If Running Long

- On slide 3, skip the reward details and only say clean training plus
  post-training surprise suite.
- On slide 7, do not read the continual-learning metrics. Say the clean column
  stays stable and remembering is 1.0.
- On slide 8, use only this sentence. The ablations show a stability versus
  peak-reward tradeoff, and the real bottleneck is trigger and data quality.

## Likely Q&A Answers

**What do PPO and IPPO mean in this project?**  
The baseline is one clean-trained actor critic checkpoint. PPO is the learning
algorithm that updates the policy. IPPO is the multi agent setup where each
drone contributes its own rollout data while sharing the same policy weights.

**Was it trained with surprises?**  
No. The clean baseline policy was trained only on `FormationAviary`. Wind,
sensor noise, sensor dropout, actuator weakness, and goal shifts were injected
after training during evaluation and adaptation.

**Why use PPO?**  
PPO is a stable actor critic policy gradient method for continuous control. It
uses clipped policy updates, which helps keep training from changing the policy
too aggressively in one update.

**Why use IPPO instead of two separate policies?**  
I wanted to test one shared swarm controller, not two unrelated controllers.
Each drone still produces its own experience, but the shared policy learns from
both drones.

**Why not use MAPPO or a centralized critic?**  
MAPPO would be a reasonable next step, but I kept IPPO because it is simpler and
matched the project goal. The main idea I wanted to test was confidence
triggered lifelong adaptation, not a new multi agent RL algorithm.

**What is the frozen baseline?**  
It is the clean checkpoint evaluated under each surprise level with no
adaptation. It answers what happens if we train once and then do nothing when
deployment conditions shift.

**What is the lifelong policy?**  
It starts from the same clean checkpoint as the frozen baseline. The difference
is that it can update between episodes when confidence drops and the episode
passes the data quality checks.

**What exactly triggers adaptation?**  
The confidence monitor combines action entropy with MC dropout variance. When
confidence drops below the threshold over the monitoring window, the trainer can
trigger an adaptation update.

**What is MC dropout doing here?**  
Dropout stays active during inference, and the policy is sampled multiple
times. If those samples disagree, the policy is less certain about what action
to take.

**Why combine entropy and MC dropout?**  
Entropy checks whether the action distribution is broad. MC dropout checks
whether the network output changes across stochastic forward passes. Using both
makes the trigger less dependent on one noisy signal.

**What is the quality gate?**  
The quality gate filters out episodes that are too short or too poor to learn
from. The point is to avoid adapting from immediate crashes or useless
trajectories.

**What is reward weighted behavior cloning?**  
It imitates actions from the adaptation buffer, but higher reward transitions
get more weight. So the update learns more from the better surprise behavior
instead of treating every transition equally.

**What is EWC?**  
EWC means Elastic Weight Consolidation. It penalizes changes to weights that
were important for the clean task, which helps reduce catastrophic forgetting.

**What is KL anchoring?**  
KL anchoring penalizes the adapted policy when it moves too far from the clean
policy distribution. It is another way to keep adaptation from drifting too far.

**What is clean replay?**  
Clean replay mixes clean task data into adaptation batches. That reminds the
policy of the original formation behavior while it learns from surprise data.

**How do the safeguards protect clean behavior?**  
EWC protects important clean weights. KL anchoring keeps the action distribution
near the clean policy. Clean replay keeps clean examples in the update. They
all push against forgetting in different ways.

**Why is clean retention such a big part of the project?**  
If adaptation improves a surprise condition but destroys clean formation flight,
then it is not really useful lifelong learning. The clean re-test checks that
the original skill survived.

**Why does moderate get worse while severe improves?**  
The adaptation trigger and episode-quality gate are conservative. Moderate can
sit in a region where the policy is degraded but does not always produce enough
useful adaptation data. Severe produces clearer low-confidence signals, but the
result is still noisy. The final three-seed validation keeps moderate as the
clearest failure case.

**Does severe improving mean severe is easier than moderate?**  
No. It means the trigger and adaptation data behaved better for severe in this
seed. Severe may produce clearer low confidence signals, while moderate can be
bad enough to hurt performance but not clear enough to trigger useful updates.

**Can we claim statistical significance?**  
No. The final report uses three controlled evaluation seeds, but all of them use
one trained checkpoint. Independent training seeds would be needed before making
a stronger statistical claim.

**Why only one seed?**  
Training time and compute were the main constraints. The evaluation path now
uses three controlled seeds, but the trained checkpoint itself is still one
clean-training seed.

**What environment values were used?**  
The main setup uses two CF2X drones in PyBullet, VEL actions, KIN observations
with waypoint information, 240 hertz physics, 30 hertz control, and 15 second
episodes with 450 control steps.

**What did the reward measure?**  
The reward encourages waypoint progress, formation spacing, target height, and
stable flight. It penalizes behavior that moves away from the intended formation
or becomes unstable.

**What does the continual learning matrix show?**  
Rows are adaptation phases, and columns are evaluation settings. The clean
column is important because it shows whether clean performance collapses after
sequential adaptation.

**What does remembering of 1.0 mean?**  
It means the best clean task performance was retained according to that metric
in the continual sequence. It supports the clean retention claim, but it does
not prove the method is optimal overall.

**Why keep safeguards if ablations sometimes get higher reward?**  
Because the objective is not only severe reward. The objective is adaptation
without forgetting. Removing safeguards can raise short-run reward but weakens
the clean-skill protection that the project is testing.

**Why does the full method have lower variance in the ablation?**  
The safeguards restrict how far the policy can move. That can limit peak reward,
but it also makes the update more stable in the severe surprise setting.

**What is the strongest result?**  
The strongest result is clean retention with partial robustness. Clean behavior
survived after adaptation, and mild and severe surprise improved in the reported
seed.

**What is the weakest result?**  
Moderate surprise got worse after lifelong adaptation. That is the clearest sign
that the trigger and adaptation objective need more work.

**What is novel here?**  
The novelty is the integrated pipeline. A post-training surprise benchmark,
dual-signal confidence trigger, between-episode adaptation with anti-forgetting
constraints, and explicit clean-after-surprise forgetting audits for the drone
swarm setting.

**Is this domain randomization?**  
No. Domain randomization would train the baseline across many randomized
conditions. Here the baseline is trained clean, and surprises are introduced
after training to test adaptation after distribution shift.

**Is this sim to real?**  
Not yet. It is a simulation study designed around post-training shift. The next
step would be stronger validation in simulation before moving toward hardware.

**How does this connect to neuroscience?**  
The connection is the idea of gated plasticity. The policy does not update all
the time. A confidence signal decides when adaptation should turn on, while the
safeguards protect older behavior from being overwritten.

**Why use drones for this project?**  
Drones make the problem concrete because small dynamics changes can matter a
lot. Wind, noisy sensing, and actuator weakness are also intuitive deployment
shifts for an adaptive controller.

**What would you improve first?**  
I would improve the adaptation trigger and data selection. The moderate result
suggests the method needs a better way to tell the difference between useful
surprise experience and experience that should not be learned from.

**What would you run next if you had more time?**  
I would run multiple seeds, increase the swarm size, compare against stronger
multi agent baselines, and test a better adaptation objective than reward
weighted behavior cloning alone.

**Could this scale beyond two drones?**  
The code structure should support larger swarms, but I would not claim scaling
from these results alone. Larger swarms would need direct experiments because
coordination and credit assignment become harder.

**Could the confidence signal be wrong?**  
Yes. Confidence is only a proxy for deployment shift. That is why the quality
gate and clean retention checks are important, and why improving calibration is
a major next step.

**What did Tanush do?**  
The technical implementation and experimental pipeline were primarily my work,
including the environment, surprise wrapper, training, confidence monitor,
adaptation loop, safeguards, evaluations, figures, and deck.

**What should I say if someone asks for the one sentence version?**  
I trained a clean two drone IPPO controller, exposed it to post-training
surprises, and tested whether confidence triggered adaptation could recover some
performance without erasing the original clean formation skill.
