Introduction
============================


AI4Science 科学与人工智能
----------------------------
介绍一下近几年 AI for science 的发展情况。
[AIDescartes]_。

介绍一下 AI for math 在 21 世纪的一些重要发展。

介绍一下 AI for physics 的发展成果，对比别人的工作以及我们的工作，总结一下它的意义。

引用一下 Marvin Minsky 的 [symbolicvsconnectionist]_，论证一下为什么符号主义、联结主义对于 AI for physics 必不可少。


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




History of Symbolic AI  符号人工智能的历史
---------------------------------------------

**Expert System 专家系统**

1965 年左右，由 Edward Albert Feigenbaum 领导的斯坦福启发式编程项目（The Heuristic Programming Project，简称 HPP）——
the Dendral project，将符号主义人工智能的研究推向了一个新的高潮。HPP 项目的主要目标是：


    ……对各种类型的科学问题以及科学和医学的各个领域中的科学推理过程的本质进行建模，从而获得深入的理解；

    ……作为方法论的一部分，并作为统筹活动，构建“专家系统”——在通常需要大量人类专业知识才能解决问题的任务上实现高水平性能的程序；因此 HPP 具有天然的应用导向。

HPP 的研究员们将人工智能看作是知识的符号表示和符号推理的计算机科学，他们认为，为了让计算机能够作为智能助手充当大部分日常工作中人类专家的角色，
计算机必须能够执行用详细的专业知识进行符号推理的任务，为了接近这个目标，研究员们提出了“专家系统”（Expert system）的概念。专家系统是模拟人类专家决策能力的计算机系统，它主要分为两个子系统：推理机和知识库——
知识库代表事实和规则；推理机将规则应用于已知事实以推断出新事实。推理引擎还可以包括解释和调试能力。

**Heuristic 启发式算法**

（这个和 Expert System 以及 Discovery 有交叠？）

介绍一下遗传算法，启发式搜索。

介绍一下 Discovery System 中运用的 Heuristic Rules。[heuretics]_

Lenat， The Nature of Heuristics, 1982。

Heuristic 和 Meta Heuristic 的概念

TODO

**Discovery Systems 发现系统**

AI 如何去自己去发现知识，维护知识库？这个过程和物理学的研究精神是一致的。

最早的一些实验：

    **AM(1976)**，[AM]_

    **Eurisko**，是对 AM 的一个继承。这两篇都是 Lenat 的文章，代码网上找得到。
    Lenat 在 1982 年以这两个 Discovery system 的工作为 case studies，总结了一些启发式编程研究的思想[heuretics]_。

    CYRANO

介绍一下 AM （Automated Mathematicians）的工作。引用一些 Lenat 论文里的话。
介绍一下它提到的困难点。

Cyc: 知识表示，本体论和知识库。（为了将这一工作推广到其他领域，Lenat 创立公司构建 Cyc 知识库。）
点评几句。

和目前 AI for math 的发展对比一下，目前的 AI for math 的工作过于关注数学的自动证明，而忽略了数学的发现过程。

这一发现过程——如何发现猜想以及如何发现新的数学概念——和 AI for physics 的精神是一致的。


[consciousness]_。




History of Connectionism  联结主义人工智能的历史
----------------------------------------------------
TODO

