\documentclass{article}

\usepackage[preprint]{neurips_2025}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{float}
\hypersetup{
  colorlinks=true,
  citecolor=black,
  linkcolor=black,
  urlcolor=blue
}

\title{Project Proposal: Confidence-Triggered Lifelong Adaptation for Drone Swarms Under Post-Training Surprises}

\author{%
  Tanush Vijayakumar Savadi \\
  UMass Amherst \\
  \texttt{tsavadi@umass.edu} \\
  \And
  Deveshi Singh \\
  UMass Amherst \\
  \texttt{deveshisingh@umass.edu} \\
  \And
  Ron Kleinhause-Goldman \\
  UMass Amherst \\
  \texttt{rkleinhauseg@umass.edu} \\
}

\begin{document}

\maketitle

\section{Background}

Drone swarms are useful because they can cover space, split work across agents, and still function if one drone fails \citep{reynolds1987boids,olfati2006flocking}. Recent multi-agent reinforcement learning methods show that decentralized swarm behavior can also be learned in simulation \citep{schulman2017ppo,dewitt2020ippo,yu2022ppo}. However, a learned policy can become brittle when the environment changes after training. In the lifelong learning setting, these changes can be seen as ``surprises'' that require the agent to notice something is wrong and then improve without forgetting earlier skills \citep{cs690nn_slides3,kirkpatrick2017ewc,zenke2017si}. This project focuses on that problem in a drone simulation setting.

\section{Problem / Opportunity / Aims}

We will build a software-only drone swarm benchmark and study post-training adaptation under controlled surprises.

\textbf{Aim 1:} Train a strong shared-policy baseline for a small drone swarm on a \textbf{formation + waypoint tracking} task in simulation \citep{panerati2021gympybullet}.

\textbf{Aim 2:} Create a reproducible surprise suite applied after training. Planned surprises include wind/drag changes, sensor noise or dropout, actuator weakening, and goal shifts \citep{cs690nn_slides3}.

\textbf{Aim 3:} Add a lightweight lifelong learning layer based on \textbf{confidence-triggered between-episode adaptation}. The main idea is to adapt only when the policy looks uncertain, while using replay and regularization to reduce forgetting \citep{kirkpatrick2017ewc,zenke2017si}. If time allows, we will also test a simple peer-help extension \citep{singh2018ic3net,cs690nn_slides3}.

Main questions:
\begin{itemize}
    \item How much does a frozen policy degrade under different surprise types and severities?
    \item Can confidence-triggered adaptation improve performance under surprise without hurting clean-task performance?
    \item If time allows, does limited peer help provide extra robustness?
\end{itemize}

\section{Methodology}

We will use \texttt{gym-pybullet-drones} as the main simulator because it supports multi-agent quadrotor control in PyBullet and is designed for RL experiments in Python \citep{panerati2021gympybullet}. To keep the setup realistic and stable, the main policy will use a \textbf{velocity-level} control interface rather than raw motor RPMs. Our main baseline will be \textbf{IPPO with parameter sharing} \citep{schulman2017ppo,dewitt2020ippo}. MAPPO will be treated as a stretch goal only if runtime and integration are manageable \citep{yu2022ppo}.

The main task will be a \textbf{two-drone formation and waypoint tracking task}. The swarm will move through a short set of waypoints while trying to keep a desired spacing between drones. Each drone will act from a local or mostly local observation containing its own state, waypoint information, and limited information about the other drone. We will first make sure the clean baseline is stable before adding surprises or adaptation.

After training, we will evaluate the frozen policy under controlled surprises such as wind changes, sensor corruption, actuator weakening, and goal shifts. The goal is to create \emph{survivable degradation} rather than immediate failure so that adaptation has useful data to learn from. The lifelong component will monitor confidence using simple signals such as policy entropy, optional MC-dropout variance, and safety-related triggers \citep{gal2016mcdropout}. If confidence drops below a threshold, the system will do small between-episode updates using recent data, replay of clean data, and anti-forgetting regularization such as EWC, with SI as an optional comparison \citep{kirkpatrick2017ewc,zenke2017si}. If time allows, we will add a simple peer-help mechanism where low-confidence drones query nearby peers under a small communication budget \citep{singh2018ic3net,cs690nn_slides3}.

For evaluation, we will compare at least:
\begin{itemize}
    \item \textbf{Frozen:} train once, then evaluate without adaptation
    \item \textbf{Lifelong:} confidence-triggered between-episode adaptation
\end{itemize}
If the peer-help extension is implemented, we will also compare \textbf{Lifelong + peers}. Metrics will include reward, waypoint completion, episode length/survival time, safety events, degradation across surprise levels, and forgetting measured as clean-task performance after adapting to surprises.

\section{Expected Outcomes, Significance, and Rationale}

We expect the clean baseline to learn the formation-and-waypoint task and then degrade under post-training surprises. We expect the lifelong version to recover some of that lost performance, especially under moderate surprises where the task is still survivable. We also expect replay and anti-forgetting regularization to reduce the drop in clean-task performance after adaptation. The main value of this project is that it connects multi-agent RL with the lifelong learning ideas from class in a concrete drone-simulation setting \citep{cs690nn_slides3}. Even if the lifelong method only partly works, the project should still show which kinds of surprise are most harmful and when adaptation becomes useful.

\section{Contingencies}

There are a few main risks. If training is too slow or unstable, we will keep the swarm size small, shorten episodes, and focus on one clean task. If the baseline is too hard to train, we will prioritize IPPO and treat MAPPO as future work. If lifelong updates are unstable, we will keep the adaptation between episodes and rely more on replay-based regularization \citep{kirkpatrick2017ewc,zenke2017si}. If peer help is too heavy to finish, we will keep confidence-triggered adaptation as the main contribution and leave peer querying/filtering as future work \citep{cs690nn_slides3}. This project is simulation-only, so full sim-to-real claims are outside the scope of the course.

\section{Timetable}

\begin{table}[H]
\centering
\caption{Planned milestones}
\begin{tabular}{ll}
\toprule
Week & Milestone \\
\midrule
1 & Finalize simulator setup and clean task design \\
2 & Train and debug IPPO baseline on clean environment \\
3 & Tune reward/task settings until the clean baseline is stable \\
4 & Implement surprise suite and measure frozen-policy degradation \\
5 & Add confidence monitoring and between-episode adaptation \\
6 & Evaluate forgetting and replay / regularization variants \\
7 & Optional peer-help extension or extra ablations \\
8 & Final plots, tables, and write-up \\
\bottomrule
\end{tabular}
\end{table}

\begin{thebibliography}{20}

\bibitem{reynolds1987boids}
Craig Reynolds.
\newblock Flocks, herds, and schools: A distributed behavioral model.
\newblock \emph{SIGGRAPH}, 1987.

\bibitem{olfati2006flocking}
Reza Olfati-Saber.
\newblock Flocking for multi-agent dynamic systems: Algorithms and theory.
\newblock \emph{IEEE Transactions on Automatic Control}, 2006.

\bibitem{panerati2021gympybullet}
Jacopo Panerati, Hehui Zheng, SiQi Zhou, James Xu, Amanda Prorok, and Angela~P. Schoellig.
\newblock Learning to fly---a gym environment with {PyBullet} physics for reinforcement learning of multi-agent quadcopter control.
\newblock \emph{IROS}, 2021.

\bibitem{song2021flightmare}
Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza.
\newblock Flightmare: A flexible quadrotor simulator.
\newblock \emph{CoRL (PMLR)}, 2021.

\bibitem{schulman2017ppo}
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
\newblock Proximal policy optimization algorithms.
\newblock arXiv:1707.06347, 2017.

\bibitem{dewitt2020ippo}
Christian Schroeder de~Witt, Tarun Gupta, and others.
\newblock Is independent learning all you need in the {StarCraft} multi-agent challenge?
\newblock arXiv:2011.09533, 2020.

\bibitem{yu2022ppo}
Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu.
\newblock The surprising effectiveness of {PPO} in cooperative multi-agent games.
\newblock \emph{NeurIPS Datasets and Benchmarks Track}, 2022.

\bibitem{orca2011}
Jur van~den Berg, Stephen~J. Guy, Jamie Snape, Ming~C. Lin, and Dinesh Manocha.
\newblock Reciprocal n-body collision avoidance.
\newblock In \emph{Robotics Research}, Springer, 2011.

\bibitem{kirkpatrick2017ewc}
James Kirkpatrick and others.
\newblock Overcoming catastrophic forgetting in neural networks.
\newblock \emph{PNAS}, 2017.

\bibitem{zenke2017si}
Friedemann Zenke, Ben Poole, and Surya Ganguli.
\newblock Continual learning through synaptic intelligence.
\newblock \emph{ICML}, 2017.

\bibitem{gal2016mcdropout}
Yarin Gal and Zoubin Ghahramani.
\newblock Dropout as a bayesian approximation: Representing model uncertainty in deep learning.
\newblock \emph{ICML}, 2016.

\bibitem{singh2018ic3net}
Amanpreet Singh, Tanmay Jain, and others.
\newblock Learning when to communicate at scale in multiagent cooperative and competitive tasks.
\newblock arXiv:1812.09755, 2018.

\bibitem{cs690nn_slides3}
Hava Siegelmann.
\newblock CS 690NN Lecture Slides (Class 3): Multi-Layer Networks; Lifelong Learning; Interacting Multi-Agent Lifelong Learning.
\newblock UMass Amherst, Feb 16, 2026.

\end{thebibliography}

\end{document}