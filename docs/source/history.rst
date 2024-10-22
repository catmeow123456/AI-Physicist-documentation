History
============================


AI4Science 科学与人工智能
----------------------------
介绍一下近几年 AI for science 的发展情况。
[AIDescartes]_。

介绍一下 AI for math 在 21 世纪的一些重要发展。

介绍一下 AI for physics 的发展成果，对比别人的工作以及我们的工作，总结一下它的意义。

引用一下 Marvin Minsky 的 [symbolicvsconnectionist]_ ，论证一下为什么符号主义、联结主义对于 AI for physics 必不可少。


    AI research must now move from its traditional focus on particular schemes. 
    There is no one best way to represent knowledge, or to solve problems, 
    and limitations of present-day machine intelligence stem largely from seeking "unified theories," 
    or trying to repair the deficiencies of theoretically neat, but conceptually impoverished ideological positions. 
    Our purely numerical connectionist networks are inherently deficient in abilities to reason well; our purely symbolic
    logical systems are inherently deficient in abilities to represent the all-important "heuristic connections” between 
    things---the uncertain, approximate, and analogical linkages that we need for making new hypotheses. 
    The versatility that we need can be found only in larger-scale architectures that can exploit and manage the advantages of several 
    types of representations at the same time. Then, each can be used to overcome the deficiencies of the others. 
    To do this, each formally neat type of knowledge representation or inference must be complemented with some "scruffier" 
    kind of machinery that can embody the heuristic connections between the knowledge itself and what we hope to do with it.




Symbolic AI  符号人工智能
---------------------------------------------

**Expert System 专家系统**

1965 年左右，以 Feigenbaum 领导的斯坦福启发式编程项目
（ The Heuristic Programming Project，简称 HPP ）
为开端，人工智能的研究走向了一个新的高潮。HPP 项目的主要目标是：

    ……对各种类型的科学问题以及科学和医学的各个领域中的科学推理过程的本质进行建模，从而获得深入的理解；

    ……作为方法论的一部分，并作为统筹活动，构建“专家系统”——在通常需要大量人类专业知识才能解决问题的任务上实现高水平性能的程序；因此 HPP 具有天然的应用导向。

HPP 的研究员们将人工智能看作是知识的符号表示和符号推理的计算机科学，
为了接近这个目标， 研究员们提出了“专家系统” （ Expert system ） 的概念。专家系统是模拟人类专家决策能力的计算机系统，它主要分为两个子系统：推理机和知识库——
知识库代表事实和规则；推理机将规则应用于已知事实以推断出新事实。

1968 年，MIT 的 MAC（the Project on Mathematics and Computation）小组启动了计算机代数项目 Macsyma。
Macsyma 可以看作是数学领域的专家系统，人们将大量数学知识以数据结构和算法的形式编码在知识库中，
通过确定的规则让程序自动化地解决各种代数、微积分、线性代数、微分方程问题等。
这一非常具有创新性的项目后来启发了 Maple、 Mathematica 等著名计算机代数系统的发展。

除了在数学领域，人们还在生物、化学、医学、法律等众多领域开发了各种各样的专家系统。

**Discovery Systems 发现系统**

专家系统的繁荣标志着第二个人工智能的夏天到来，人们梦想着设计出试图发现新的科学概念或规律的人工智能系统，
它被叫做“发现系统”（Discovery System）。
理想情况下，发现系统应该能够提出有趣的科学概念，
系统地在假设（也就是猜想）空间中搜索寻找概念之间可能存在的关系，
并产生最能描述数据中复杂模式的猜想。

    There exist too many combinations to consider all combinations of existing entities;
    the creative mind must only propose those of potential interest.
    —— Poincare

1976 年，Lenat 发表了他有关 AI 数学家（Automated Mathematician，简称 AM） [AM]_ 的开创性工作，
荣获两年一度的 IJCAI 计算机与思想奖。
在 AM 程序中，最重要的是名为“概念”的数据结构和“启发式规则”（Heuristic）。
从最初的数百个概念和启发式规则出发，AM 通过一个简单的控制流程，
不断地生成新的概念，从而解决各种数学问题。
AM 程序从集合论的第一原理出发，发现了偶数、乘法、素数等概念，并发现了算数基本定理以及哥德巴赫猜想；
然后随着程序的持续运转，AM 发现的概念数量迅速膨胀产生大量无意义知识，
而没能发现取模、指数等概念，因此 AM 程序没能在数论领域取得进一步突破。


Lenat 随后投入了 Eurisko [Eurisko]_ 的开发，采用了一种名为 RLL-1 的表示语言，
将 AM 的启发式搜寻概念的范式扩展到了单一的数学领域以外；
Eurisko 所使用的启发式规则还包括了描述如何使用和更改启发式规则的规则 [heuretics]_，
这使得 Eurisko 中启发式规则的创建和修改变得更加简单和灵活。
Eurisko 被应用于许多领域并取得了惊人的成功，其中包括 VLSI 芯片设计。
Lenat 在设计 Eurisko 程序时为编码领域知识耗费了巨大的精力，
因此他决定构建一个全面的本体论和知识库。
1984 年，Lenat 启动了 Cyc 项目，旨在以机器可用的形式编纂构成人类常识的数百万条知识，
Cyc 项目对人工智能领域的本体论工程以及知识表示有着深远的影响。

.. CYRANO
.. 和目前 AI for math 的发展对比一下，目前的 AI for math 的工作过于关注数学的自动证明，而忽略了数学的发现过程。
.. 这一发现过程——如何发现猜想以及如何发现新的数学概念——和 AI for physics 的精神是一致的。

八十年代末期，专家系统计算机的市场崩溃了，伴随着人工智能的
第二个冬天的到来，“专家系统”、“发现系统”等词语逐渐淡出了大部分人的视野。
随后人工智能诞生了各种新的形态，以人工神经网络为代表的联结主义方法开始复兴，
人们将专家系统中的一些思想迁移到深度学习中，提出了“持续学习” （ Continuous Learning）的概念。
持续学习希望模型能够接连不断地学习多个任务的数据，并且不断地积累从任务中学到的知识，过去任务的经验能够使得模型在新任务上的表现更好。
持续学习关注的核心问题是灾难性遗忘 （ Catastrophic Forgetting ） ，大多数神经网络只能学习单一的任务，
或者在学习新任务过后无法再保持在旧任务上的效果。因此记忆并维护知识库、设计各种防遗忘机制是持续学习的主要目标。

符号主义人工智能也发生了变革，人们开始关注如何增强知识库的推理能力。形式化验证与自动推理研究开始兴起，
这一块主要在“机器辅助证明”小节 :ref:`proofassistant` 介绍。

近年来，发现系统再度进入人们的视野。2007 年，由 Schmidt 与他的导师 Lipson 一起开发的 Eureqa 软件
使用遗传算法来搜索数学方程，从而找到描述数据集的最简单的形式（这种方法被称为符号回归）。
在 2011 年 10 月版的《Physical Biology》中，Lipson 描述了一项酵母实验，该实验预测了七个已知方程。
这项工作有效地证明了符号回归是人工智能驱动的科学发现的一个可行方向。
21 世纪，AI 在科学发现领域的研究逐渐兴起人们给这一方向赋予了一个新的名词：AI for Science。
除了在生物、化学、医药、材料等领域的迅速突破，一些研究者还开始尝试将 AI 应用于物理学领域，
AI Feynman、 AI descarte [AIDescartes]_ 等项目在这个方向上取得了一些成果，人们开始期待着让
AI 自动从数学中挖掘出物理公式，甚至提取出一般化的物理定律。

Miles Cranmer 及其团队在 2023 年开发 PySR `<https://astroautomata.com/PySR/>`_ ，
这是一个用于符号回归的 Python 库。
IBM 团队在 2024 年开发了 AI Hilbert `<https://ai-hilbert.github.io/>`_ ，该系统在具体物理问题中，自动发现数据所满足的物理定律，
并基于一个计算代数几何的算法来做符号公式的验证和推理，判断公式是否与公理集合相容。

与 AI for Science 领域的符号回归方法对应的是 Program Synthesis 程序合成。
一些人尝试让 AI 去学习合成一段程序来给出一个数列的通项公式，相关的工作有
Learning Program Synthesis for Integer Sequences
from Scratch `<https://arxiv.org/pdf/2202.11908>`_ ，相关的开源程序为 `<https://github.com/barakeel/oeis-synthesis.git>`_ ，
其论文声称在不提供人工标注的数据的情况下，AI 通过 self-learning ，可以在 OEIS 数据集上发现 27987 个序列的通项公式。


**Heuristics 启发式方法**

介绍一下遗传算法，启发式搜索。

介绍一下 Discovery System 中运用的 Heuristic Rules。[heuretics]_

Lenat, The Nature of Heuristics, 1982。

Heuristic 和 Meta Heuristic 的概念

TODO


[consciousness]_。

.. _proofassistant:

Proof Assistant/Checker 机器辅助证明
---------------------------------------------
1921 年， Hilbert 提出了希尔伯特纲领（ Hilbert Program ），它要求以公理化的形式将所有数学形式化，并证明这种数学公理化是一致的。
所谓的“形式化”， 是指所有数学应该用一种统一的严格形式化的语言， 并且按照一套严格的规则来使用。Hilbert 希望能够为全部的数学提供一个安全的理论基础，
具体地，这个基础包括

- 完备性：在形式化后，数学的每一个命题都能够被证明或证伪；
- 一致性：运用这一套形式化和它的规则，不可能推导出矛盾（也就是说不存在一个命题，它既能被证明又能被证伪）；
- 可判定性：存在一个算法，能够判定每一个形式化的命题是真还是假。

1931 年 Gödel 提出了哥德尔的不完备性定理（ Gödel's incompleteness theorems ）。
Gödel 证明了，任何一个形式系统，只要包括了简单的初等数论描述，而且是一致的，则一定存在一个命题既不能被证明也不能证伪。
Gödel 的论文展示了定理证明、计算、人工智能、逻辑和数学本身的基础局限性（有些人误解了他的结果，以为他证明的是人类优于 AI ），在学术界引起了轰动，
这一研究对 20 世纪基础数学和哲学的发展产生了巨大影响，也奠定了理论计算机科学和人工智能理论的基础。
1940 年代至 70 年代的大部分人工智能和定理证明有关，大多是通过专家系统和逻辑编程进行 Gödel 式的定理证明和演绎。[godel]_

1935 年， Alonzo Church 设计了一种通用编码语言（ Untyped Lambda Calculus ），并运用它推导出哥德尔结果的推论，
这门语言构成了极具影响力的编程语言 LISP 的基础。 1936 年， Alan Turing 引入了另一个通用模型“图灵机”（ the Turing Machine），
它是计算机科学领域最著名的模型之一，成为了后来通用可编程的电子计算机的理论基础。

1940 年， Alonzo Church 基于 Lambda 演算提出了简单类型论（ simple type theory ） [typetheory]_。在类型论中，一切元素首先归属于某个类型，而后才能开始讨论其性质，
由此可以解决 Russell 悖论。在类型论的观点下，一切数学对象（如整数，实数，群，拓扑空间）的含义都由它们从属的类型决定。
例如函数 :math:`\lambda x. f x` 代表函数 :math:`x \mapsto f x`，如果 :math:`x` 是类型 :math:`s_1` 的元素， :math:`f x` 是类型 :math:`s_2` 的元素，
那么函数 :math:`x \mapsto f x` 就是类型为 :math:`s_1 \to s_2` 的元素。可以在这个类型论中加入更多的符号用来表达更复杂的数学对象，
例如用归纳定义的 :math:`0,S(0),S(S(0)),\ldots` 表示自然数，用 :math:`\wedge` 表示且命题，:math:`\vee` 表示或命题，用 :math:`\neg` 表示非命题，用 :math:`\forall` 表示全称量词等等。
那么数学归纳法就可以被表达为 

.. math::

    \forall(\lambda P.P(0) \wedge \forall(\lambda n.P(n) \Rightarrow P(S(n))) \Rightarrow \forall(\lambda n.P(n)))

人们依据这种形式系统, 
写出了计算机程序 Isabelle（由 Lawrence C. Paulson 及其团队和 Tobias Nipkow 于 1986 完成），用计算机验证数学证明, 从而达到极高的准确性。

Church 的 Typed Lambda Calculus 启发了后来的 Curry-Howard correspondence。人们发现, 用于计算、为数学对象分类的类型, 与用于证明、只有真假的命题，
在许多方面有着惊人的相似性 —— 命题是类型，而其对应的证明是从属于该类型的一个元素。
Curry-Howard correspondence 揭示了逻辑学和类型论（进而与计算机科学）之间有着深刻的内在联系。

1970 年代, Martin-Löf 提出了一种构造主义的类型论（Martin-Löf type theory, 缩写为 MLTT）。
人们基于 Martin-Löf 类型论及其变体设计了许多交互式定理证明器（Interactive Theorem Prover, ITP），如 agda, coq, lean 等。这些定理证明助手可以用来验证数学定理，编写程序，甚至证明计算机程序的正确性。
在使用 Coq 时，每当用户输入一个证明，Coq 会时刻告诉用户当前这一步有哪些条件，目前还有哪些目标，用户可以使用一些 Coq 提供的证明策略（tactics）来推进对目标的证明。

在 2005 年, Georges Gonthier 等人在 Coq 中完全形式化了四色定理的证明。这个定理目前人类所知的证明中涉及到了上千种情况的讨论, 因此人力几乎不可能保证其正确无误。
2009 年, Xavier Leroy 等人开 发了完全经由 Coq 验证的代码编译器 CompCert，证明了这样规模的形式化验证在实践中是可行的.

Lean 最初是由 MSR 的 Leonardo de Moura（Z3 作者）开发的一个实验性项目，
Lean 的社区的主要目标是构建一套完整的数学定理库（Mathlib），以及便于自动化的开发（用户可以编写自己的 Tactics 来指引定理证明器构造证明项）。
2017 年 Lean 3 诞生，由于支持了元编程（meta programming）框架，用户可以用 Lean 的语言来操作 Lean 的表达式，进而编写 Lean 的元程序。
越来越多的数学家开始投入到 Mathlib 库的开发和维护中 `<https://github.com/leanprover-community/mathlib4>`_，越来越多的现代数学中重要的数学定理被形式化。

最近的几年里，自动定理证明（automated theorem proving，ATP）领域通过引入深度学习和强化学习，
尤其是凭借大语言模型（Large language model, LLM）的飞速进展，产生了大量令人惊叹的工作，如
Advancing mathematics by guiding human intuition with AI `<https://www.nature.com/articles/s41586-021-04086-x>`_，
Alpha Geometry `<https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/>`_。
研究者们还发起了 IMO Grand Challenge `<https://imo-grand-challenge.github.io/>`_，希望让AI拿到IMO金牌。
Deepmind 的 Alpha Proof 项目 `<https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/>`_ 
在 IMO 的几道题目上取得了银牌水平的成绩（在代数与数论相关的问题上，AI 正确地写出了 Lean 的证明）。


MetaPhysics 元物理学（不确定要不要加这章，有点偏哲学）
--------------------------------------------------------
TODO
https://mally.stanford.edu/

Connectionism  联结主义人工智能
----------------------------------------------------

**Neural Networks 神经网络与反向传播**

**LLM 大语言模型**

**Reinforcement Learning 强化学习**

TODO

