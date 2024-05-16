Specific Model 具体模型
==============================

这一章主要介绍一下 Specific Model 在数学上的定义和目标，以及它在代数化简、排除冗余层面的原理。

Representation of a Differential Equation System 微分方程组的表示
--------------------------------------------------------------------------------------------------

**Rep. of Diff.Eq.System 微分方程组的表示**

对于一个微分方程组 :math:`\{a_1, a_2, \cdots, a_m\}`，它的某个子集

.. math::

    S = \{a_{i_1}, a_{i_2}, \cdots, a_{i_n}\}, 1\le i_j \le m (1\le j \le n)

如果满足其他所有微分方程都出现在它所生成的根微分理想中，那么称 :math:`S` 是微分方程组的一个表示。

**Minimal Rep. of Diff.Eq.System 微分方程组的极小表示**

对于一个微分方程组的一个表示，如果它的任意真子集都不是一个表示，
那么称它是一个极小表示。

这样的极小表示可能有很多个。为了在 Specific Model 中保留足够简洁的极小表示，
我们需要具体地定义表达式的复杂度，复杂度是对一个表达式的复杂程度的度量，
由于“简洁”与“复杂”是非常主观的概念，所以对复杂程度的定义并没有一个唯一的标准。
只要定义得合理，不同的复杂度定义方式对程序运行结果并没有影响。下面给出一个可行的复杂度定义：

**complexity of an expression 表达式的复杂度**

下面我们将复杂度函数简写为 :math:`\text{complexity}(A) \rightarrow f(A)`。  

1. 基本概念的复杂度为其下标个数 :math:`+1`。例如 :math:`f(t) = 1, f(m_i) = 2`。  

2. :math:`n` 阶导数的复杂度为被求导表达式复杂度 :math:`+n`。例如 :math:`f(\partial_t^n A) = f(A) + n`。  

3. 概念的复杂度为其定义式的复杂度。例如 :math:`f(p_i) = f(m_i v_i), f(v_i) = f(\partial_t x_i)`。  

4. 由概念组成的表达式的复杂度为，所有概念的复杂度之和。例如

.. math::

    f\left(A + \frac{B C}{D^2}\right) = f(A) + f(B) + f(C) + 2 f(D) + 2

5. 不同表达式具有相同复杂度，为了进一步比较它们复杂度的大小，可以将复杂度扩大为元组（tuple）： :math:`\text{complexity}(A) \rightarrow (f(A), \text{len}(A))`，因此 :math:`\text{complexity}(p_i) < \text{complexity}(m_i v_i)`。

**Regular Rep. of Diff.Eq.System 微分方程组的正则表示**

一个微分方程组的正则表示是指它的 `complexity` 字典序最小的一个表示（不一定是极小表示），

**求微分方程组的正则表示**

维护一个集合 :math:`S`，并初始化一个微分代数对象 `diffalg`。
将 :math:`\{a_1, a_2, \cdots, a_m\}` 按照 `complexity` 从小到大排序，
然后把它们的依次添加到 `diffalg` 中：

1. 如果当前添加的方程的 `rawdefinition` 不属于之前方程所生成的根微分理想，则添加成功，并将当前方程添加到集合 :math:`S` 中，并将它的 `rawdefinition` 用于更新新的根微分理想。

2. 否则当前方程就被认为是冗余的方程，不添加。

这样得到的集合 :math:`S` 一定是字典序最小的，所以就是微分方程组的正则表示。

使用 `diffalg.insert_new_eq` 不断地添加方程，最终可以通过 `diffalg.eqs` 得到正则表示。
一个例子：

.. code-block:: python
    :emphasize-lines: 14,15,16,17,18

    import sympy as sp
    from aiphysicist.diffalg import DifferentialRing, diffalg
    t, k = sp.symbols('t k')
    pos = sp.Function('pos')(t)
    posr = sp.Function('posr')(t)
    ring = DifferentialRing([('lex', [pos, posr]),
                             ('lex', [k])])
    pos, posr = sp.symbols('pos posr')
    eq1 = pos - posr
    eq2 = sp.Derivative(pos, t) - sp.Derivative(posr, t)
    eq3 = k - sp.Derivative(pos, t) / sp.Derivative(pos, t, 3)
    eq4 = sp.Derivative(posr, t) / k - sp.Derivative(posr, t, 3)

    ideal = diffalg(ring)
    ideal = ideal.insert_new_eq(eq1)
    ideal = ideal.insert_new_eq(eq2)
    ideal = ideal.insert_new_eq(eq3)
    ideal = ideal.insert_new_eq(eq4)

>>> print(ideal.eqs)
[pos - posr, k - Derivative(pos, t)/Derivative(pos, (t, 3))]

另一个例子：

.. code-block:: python
    :emphasize-lines: 12,13,14,15

    import sympy as sp
    from aiphysicist.diffalg import DifferentialRing, diffalg
    t = sp.symbols('t')
    x = sp.Function('x')(t)
    y = sp.Function('y')(t)
    z = sp.Function('z')(t)
    ring = DifferentialRing.default([x, y, z])
    x, y, z = sp.symbols('x y z')
    eq1 = sp.Derivative(y, t, 2)
    eq2 = y * sp.Derivative(x, t, 2)
    eq3 = x * sp.Derivative(y, t) - y * sp.Derivative(x, t)
    ideal = diffalg(ring)
    ideal = ideal.insert_new_eq(eq1)
    ideal = ideal.insert_new_eq(eq2)
    ideal = ideal.insert_new_eq(eq3)

>>> print(ideal.eqs)
[Derivative(y, (t, 2)), y*Derivative(x, (t, 2)), x*Derivative(y, t) - y*Derivative(x, t)]
>>> ideal = diffalg.from_eqs(ring, [eq1, eq2])
>>> print(ideal.belongs_to(eq3))
False
>>> ideal = diffalg.from_eqs(ring, [eq1, eq3])
>>> print(ideal.belongs_to(eq2))
True

可以看到 `ideal.eqs = [eq1, eq2, eq3]` 并不是微分方程组的极小表示。
`eq2` 实际上位于 `[eq1,eq3]` 生成的根微分理想中。
正则表示不一定是微分方程组的极小表示，但是我们可以再从后往前依次作一遍检查，
如果某个方程位于除它以外其他方程所生成的根微分理想中，就将它从集合中删去，那么我们最终得到的才是
微分方程组的极小表示。

**求 complexity 字典序最小的极小表示**

初始化集合 :math:`S` 为微分方程组中所有的方程 :math:`\{a_1, a_2, \cdots, a_m\}`，
并将这些方程按照 :math:`complexity` 从小到大排序，并从后往前枚举这些方程，
设当前枚举到的方程是 :math:`a_i`：

1. 对 :math:`S - \{a_i\}` （的 `rawdefinition`）构建临时微分代数对象 `diffalg_temp`，得到它们的根微分理想的正则分解表示。如果 :math:`a_i` 的 `rawdefinition` 属于 `diffalg_temp` 的根微分理想，那么 :math:`a_i` 就被认为是冗余的，将它从 :math:`S` 中删去。

2. 否则，继续保持 :math:`a_i` 在 :math:`S` 中。

这样得到的集合 :math:`S` 就是微分方程组的极小表示。

*求微分方程组的极小表示的时间代价较高且没有必要，因此在我们的程序实现中，Specific Model 保留的是微分方程组的正则表示。*

Representation of Conserved Quantities 守恒量集合的表示
--------------------------------------------------------------------------------------------------

在一次循环的工作流程中，理论家会发现一些表达式的值是守恒的，它被称为守恒量。
守恒量是实验运动数据的一个重要性质，因为它是对实验数据所满足的规律的一个总结，
而且由于其表达式是由概念构成的，它往往是可迁移的（在其他实验中这样的表达式也可能有意义），
所以我们有动机将它定义为新的概念。

当我们将守恒的表达式 :math:`eq_i`
添加到 Specific Model 中，我们赋予它一个常量名 :math:`C_i`。
为了防止在不停的循环过程中守恒量集合无限制地膨胀，我们需要设计一个算法，将守恒量集合简化为它的极小表示。

除了非平凡的守恒表达式以外，还存在一些平凡的常量（它们对应当前实验中每一物理对象的内禀概念，本来就是不随时间变化的量）。设
这些平凡常量构成的集合记为 :math:`I_{\text{Const}}`。


**Minimal Rep. of CQs 守恒量集合的极小表示**

:math:`C_i` 包括平凡常量在内可能存在某种依赖关系：

.. math::

    f(C_{i_1}, C_{i_2}, \cdots, C_{i_m}, I_{\text{Const}}) = 0

这样的关系在改变任何实验参数的情况下总是成立。换句话说，
在实验物理对象确定，切其中 :math:`m-1` 个常量的值已知的情况下，就能唯一确定剩下的那个常量。
所以常量集合 :math:`\{C_{i_1}, \cdots, C_{i_m}\}`
是非独立的，我们称这个常量集合中存在冗余的常量。不断地删除冗余常量，直到不存在冗余常量为止，
最终得到的常量集合就被称为极小表示。我们将常量集合 :math:`\{C_1,\cdots, C_n\}` 的极小表示 :math:`S` 的大小称为
这组常量集合的“维度”，记作 :math:`\dim\{C_1,\cdots, C_n\}`。极小表示 :math:`S` 满足 :math:`\dim S = |S|`。

例如：匀加速直线运动里，理论家发现了三个守恒量：加速度、初速度、初位置。分别定义三个常量：

.. math::

    &C_1 = a(t)\\
    &C_2 = v(t) - a(t) t\\
    &C_3 = x(t) - v(t) t + \frac{1}{2} a(t) t^2

这三个常量是互相独立的，因此这个守恒量集合的极小表示就是这三个常量。

在碰撞实验中，理论家发现了三个守恒量，分别对应总动量、总能量、速度差的平方：

.. math::

    &C_1 = m_1 v_1(t) + m_2 v_2(t)\\
    &C_2 = m_1 v_1(t)^2 + m_2 v_2(t)^2\\
    &C_3 = (v_1(t) - v_2(t))^2

这三个常量是非独立的，它们之间存在一条代数关系 :math:`2 C_2 m_1 + 2 C_2 m_2 - C_1^2 - C_3 \cdot m_1 \cdot m_2 = 0`。
因此这个守恒量集合的极小表示为 :math:`\{C_1, C_2\}$（或 $\{C_1, C_3\}$, $\{C_2, C_3\}`）。

在弹簧实验中，弹簧左端固定，右端连接质点，质点作振荡运动。
:math:`pos=posr` 为质点的空间坐标（弹簧右端点坐标）。理论家发现了五个守恒量：

.. math::

    &C_1 = posl(t) \\
    &C_2 = mv(t)^2 + k \Delta L(t) ^2 \\
    &C_3 = \frac{\partial_t^3 pos(t)}{\partial_t pos(t)}\\
    &C_4 = ma(t) + k \cdot pos(t)\\
    &C_5 = - \partial_t^3 pos(t)\partial_t pos(t) + (\partial_t^2 pos(t))^2

这五个常量之间存在这几条代数关系： 

.. math::

    C_4 - k (C_1 + freeL) = 0,\quad C_3\cdot m + k = 0,\quad -C_5 m^2 + C_2 k = 0

这个守恒量集合的极小表示为 :math:`\{C_1, C_2\}` （或 :math:`\{C_4, C_2\}, \{C_1, C_5\}, \{C_4, C_5\}` ）。
实际上弹簧实验还存在一个隐蔽的守恒量：初始相位。但由于构造这一守恒量需要三角函数相关的符号，所以超出了我们目前 AI 理论家的能力范围。


**求 complexity 字典序最小的极小表示**

对于守恒量表达式 :math:`eq_1, eq_2, \cdots, eq_n`，分别对应常量名 :math:`C_1, \cdots, C_n`。
在固定当前实验中物理对象不变的条件下，调节不同的实验参数进行 :math:`N` 次实验，
计算得到 :math:`C_1, C_2, \cdots, C_n` 的 :math:`N` 组取值，
可以对应 :math:`n` 维空间中的 :math:`N` 个点。这 :math:`N` 个点所构成的流形的维数 :math:`\dim\{C_1,\cdots, C_n\}` 实际上就是独立无关的守恒量的个数，
也就是极小表示的大小。可以用 scikit-dimension 估计点集所构成流形的维度，点集大小的数量级应为
:math:`N\sim 10^n`。

不妨设这些常量已经按照 `complexity` 从小到大排序，维护一个集合 :math:`S`，
从前往后枚举这些常量，设当前枚举到的常量是 :math:`C_i`：

1. 如果 :math:`\dim (S \cup \{C_i\}) = \dim S + 1`，那么常量 :math:`C_i` 与集合 :math:`S` 是独立的，将它添加到 :math:`S` 中。

2. 如果 :math:`\dim (S \cup \{C_i\}) = \dim S`，那么常量 :math:`C_i` 与集合 :math:`S` 是相关的，将它视作是冗余的，不添加。

最终得到的常量集合 :math:`S` 就是守恒量集合的极小表示。

**Regular Rep. of CQs 守恒量集合的正则表示**

在程序实现中，如果常量 :math:`C_1, C_2, \cdots, C_m` 之间的关系

.. math::

    f(C_1, C_2, \cdots, C_m, I_\text{Const}) = f(eq_1, eq_2, \cdots, eq_m, I_\text{Const}) = 0

已经被包含在微分代数对象 `diffalg` 的根微分理想中，那么它们之间的就存在一条解析的代数关系，这条代数关系已经被理论家发现。
我们称这种冗余为“解析的冗余”。将所有理论家发现的解析的冗余排除掉后得到的常量集合被称为是正则表示。

**求守恒量集合的正则表示**

对于守恒量表达式 :math:`eq_1, eq_2, \cdots, eq_n`，分别对应常量名 :math:`C_1, C_2, \cdots, C_n`。
初始化常量集合 :math:`S = \{C_1, C_2, \cdots, C_n\}`，
在当前 Specific Model 的 `diffalg` 的基础上，添加若干条带常量的方程 :math:`C_i - eq_i = 0\ (1\le i\le n)`。
并调整变元排序，设置三个 block。第一个 block 由普通微分变元构成；
第二个 block 由常量 :math:`C_i (1\le i\le n)` 构成，并按照 `complexity` 从大到小排序；
第三个 block 为平凡常量 :math:`I_{\text{Const}}`。

在这样的变元排序下，Rosenfeld Groebner 算法得到的正则微分链包含了常量之间可能存在的代数关系，
遍历其中每一条关系，如果一条关系完全由常量组成，那么将其中 `complexity` 最高的 :math:`C_i` 视作是解析冗余的，从集合 :math:`S` 中删去。
这样得到的集合 :math:`S` 就是守恒量集合的正则表示。

在程序实现中要注意的是，如果删去了冗余常量 :math:`C_i`，那么就需要将 `diffalg` 中所有出现 :math:`C_i` 的地方替换为 :math:`eq_i` 再进行一次更新。


Definition of Concept and Packaging of Concept 概念的定义与包装
-----------------------------------------------------------------------------------------------

TODO

DOF and CQs in Dynamical systems 动力学系统的自由度与守恒量
-----------------------------------------------------------
在物理学中，自由度（degree of freedom，简写为 DOF）是指描述一个系统的最小独立参数的数量。
当我们聚焦于一个实验，讨论一个它所描述的系统的自由度时，实际上有两种考察方式：

1. 从微分方程组的角度考察：
一个系统的自由度是指它的微分方程组的自由度（独立的初始条件的个数）。此时我们对系统自由度的定义依赖于我们对它的认识（我们所构建的理想模型）。

2. 从实验数据的角度考虑：
一个系统的自由度是指实验真实数据的自由度（所依赖的独立参数的个数）。

下面我们将介绍求这两种自由度的方法。

**求实验数据的自由度**

在固定物理对象不变的条件下改变实验的输入参数生成大量的实验数据。
假设所有的观测数据的长度都为 :math:`n_t`，每次实验获得的一组实验数据包含 :math:`k` 个观测数据，
那么总共就有 :math:`n_t\cdot k` 个浮点数，一组实验数据对应于 :math:`n_t\cdot k` 维空间中的一个点。

假设总共生成了 :math:`N` 组实验数据，当 :math:`N` 趋向于无穷大时，这 :math:`N` 个点将构成一个 :math:`n_t\cdot k` 维空间中一个稠密的点集，
它所反映的流形的维数就是实验数据的自由度。由于 :math:`n_t\cdot k` 太大，上述做法并不可行。
可以将 :math:`n_t\cdot k` 维空间投影到一个更低维的 :math:`m` 维空间中（左乘一个 :math:`m \times (n_t\cdot k)` 的随机矩阵），
再对 :math:`m` 维空间中的点集的流形维度进行估计。
python 的 scikit-dimension 库提供了对数据点所构成流形维度进行估计的方法。
假设实验数据的真实自由度为 :math:`d`，为了取得更精确的估计，所选择的参数 :math:`m, N` 需要满足 :math:`d < m, N \sim 10^m`。

**求微分方程的自由度**

微分方程组的自由度就是它独立的初始条件的个数。由于求实验数据的自由度需要大量的实验数据，所以在程序实现和理论分析中，我们更关注微分方程的自由度，
它虽然不一定反映了实验数据的真实自由度，但它是对理论家所发现的微分方程组的进一步认识，与微分方程的求解、
系统随时间演化的行为有着密切联系。

先求出微分方程组的正则微分链表示，记为 `regchain`，
然后调用 maple 的 DifferentialAlgebra 库中的
`Get(initialconditions, n, regchain)` 方法。 
例如，考察一个弹簧连接两个质点的实验：

.. code-block:: text

    interface(prettyprint=0):
    with(DifferentialAlgebra):
    x1 := pos1(t);
    x2 := pos2(t);
    v1 := diff(x1, t);
    v2 := diff(x2, t);
    a1 := diff(x1, t$2);
    a2 := diff(x2, t$2);
    eqs := [
        m1 * a1 - k * (x2 - x1 - L),
        m2 * a2 + k * (x2 - x1 - L)
    ];
    R := DifferentialRing(blocks = [[pos1, pos2], [m1, m2, k, L]], derivations = [t]);
    ideal := RosenfeldGroebner(eqs, R);
    regchain := ideal[1];

>>> print(Equations(regchain));
[k*L+m1*diff(pos1(t),t $ 2)-k*pos2(t)+k*pos1(t), -k*L+k*pos2(t)-k*pos1(t)+m2*diff(pos2(t),t $ 2)]
>>> Get(initialconditions, 10, regchain);
{L, k, m1, m2, diff(pos1(t),t), diff(pos2(t),t), pos1(t), pos2(t)}

将平凡常量 :math:`L, k, m_1, m_2` 去除后，得到的集合

.. math::

    \{\partial_t pos_1(0), \partial_t pos_2(0), pos_1(0), pos_2(0)\}

就是系统的独立的初始条件，因此微分方程的自由度为 :math:`4`。可以看到初始条件中不包含二阶及以上的导数，
这样的动力学系统在数学上被称为哈密顿系统，系统的演化由相空间中的相流（每个点都对应一条相曲线）决定。
自由度为 :math:`n` 的哈密顿系统的相空间是 :math:`2n` 维的（在数学上是具有辛形式 :math:`\omega` 的辛流形），决定该系统演化的微分方程
的自由度为 :math:`2n`，因为它具有 :math:`2n` 个独立的初始条件。

下面再考察一个带约束的系统，轻杆连接两个质点在二维平面内作自由运动：

.. code-block:: text

    interface(prettyprint=0):
    with(DifferentialAlgebra):
    x1 := pos1x(t);
    y1 := pos1y(t);
    x2 := pos2x(t);
    y2 := pos2y(t);
    vx1 := diff(x1, t);
    vy1 := diff(y1, t);
    vx2 := diff(x2, t);
    vy2 := diff(y2, t);
    eqs := [
        (x1 - x2)^2 + (y1 - y2)^2 - L^2,
        m1 * vx1 + m2 * vx2 - PX,
        m1 * vy1 + m2 * vy2 - PY,
        PX <> 0, PY <> 0,
        diff(m1 * (x1 * vy1 - y1 * vx1) + m2 * (x2 * vy2 - y2 * vx2), t),
        m1 <> 0, m2 <> 0, L <> 0, m1+m2 <> 0, pos1x(t)-pos2x(t) <> 0, pos1y(t)-pos2y(t) <> 0
    ];
    R := DifferentialRing(blocks = [[pos1x, pos1y, pos2x, pos2y], [PX, PY], [L, m1, m2]], derivations = [t]);
    ideal := RosenfeldGroebner(eqs, R);
    regchain := ideal[1];

>>> Get(initialconditions, 10, regchain);
{L, PX, PY, m1, m2, diff(pos2y(t),t), pos1y(t), pos2x(t), pos2y(t)}


将平凡常量 :math:`L, m_1, m_2` 去除后，得到的集合

.. math::

    \{P_X, P_Y, \partial_t pos_{2y}(0), pos_{1y}(0), pos_{2x}(0), pos_{2y}(0)\}

就是系统的独立的初始条件。因此微分方程的自由度为 :math:`6`。

考察开普勒问题，二维平面内一个粒子在中心势 :math:`F(r) = - k/r` 下运动：

.. math::

    &x(t)^2 + y(t)^2 - r(t)^2 = 0\\
    &\frac{1}{2} m \left(v_x(t)^2 + v_y(t)^2\right) - \frac{k}{r(t)} - E = 0\\
    &(x(t) p_y(t) - y(t) p_x(t)) - J_z = 0\\
    &p_y(t) J_z - m k \frac{x(t)}{r(t)} - A_x\\
    &-p_x(t) J_z - m k \frac{y(t)}{r(t)} - A_y

其中 :math:`v_x(t) \rightarrow \partial_t x(t), v_y(t) \rightarrow \partial_t y(t), p_x \rightarrow m v_x(t), p_y \rightarrow m v_y(t)`。
:math:`E` 为总能量，:math:`J_z` 为垂直于平面的角动量分量，:math:`A_x, A_y` 为龙格-楞次矢量（Runge-Lenz Vector）的两个分量。
考察微分方程组的独立的初始条件，将包含龙格-楞次方程和不包含龙格-楞次方程的情况作对比：

.. code-block:: text

    interface(prettyprint=0):
    with(DifferentialAlgebra):
    x := posx(t);
    y := posy(t);
    r := dist(t);
    vx := diff(x, t); vy := diff(y, t);
    px := m * vx; py := m * vy;
    eqs := [
        x^2 + y^2 - r^2,
        m * (vx^2 + vy^2) / 2 - k / r - E,
        (x * py - y * px) - Jz,
        k <> 0, Jz <> 0, m <> 0, r <> 0, diff(r, t) <> 0, E <> 0
    ];
    R := DifferentialRing(blocks = [dist, [posx, posy], [E, Jz], [k, m]], derivations = [t]);
    ideal := RosenfeldGroebner(eqs, R);
    regchain := ideal[1];

>>> Get(initialconditions, 10, regchain);
{E, Jz, k, m, posx(t), posy(t)}

.. code-block:: text

    interface(prettyprint=0):
    with(DifferentialAlgebra):
    x := posx(t);
    y := posy(t);
    r := dist(t);
    vx := diff(x, t); vy := diff(y, t);
    px := m * vx; py := m * vy;
    eqs := [
        x^2 + y^2 - r^2,
        m * (vx^2 + vy^2) / 2 - k / r - E,
        (x * py - y * px) - Jz,
        py * Jz - m * k * x / r - Ax,
        -px * Jz - m * k * y / r - Ay,
        k <> 0, Jz <> 0, m <> 0, r <> 0, diff(r, t) <> 0, E <> 0, k^2*m^2-Ax^2 <> 0, Ax <> 0, Ay <> 0
    ];
    R := DifferentialRing(blocks = [dist, [posx, posy], [E, Jz, Ax, Ay], [k, m]], derivations = [t]);
    ideal := RosenfeldGroebner(eqs, R);
    regchain := ideal[1];

>>> Get(initialconditions, 10, regchain);
{Ax, Ay, Jz, k, m, posy(t)}

可以看到，不包含龙格-楞次方程的情况下，微分方程组的独立初始条件为

.. math::

    E, J_z, x(0), y(0)

加入龙格-楞次方程后，
微分方程组的独立初始条件为

.. math::

    J_z, A_x, A_y, y(0)

通过引入龙格-楞次矢量，开普勒运动的轨道形状完全被守恒量决定，
这体现在微分方程组的 :math:`4` 个独立的初始条件中有 :math:`3` 个是守恒量，
这 :math:`3` 个独立的守恒量将 :math:`4` 维相空间中的运动限制在一条曲线上，
剩下一个初始条件 :math:`y(0)` 决定了运动的起点。

如果将这一动力学系统当作哈密顿系统来研究，则可以挖掘出更多的数学信息。
在力学发展早期，人们的一个重要任务就是为给定的哈密顿系统寻找各种运动积分（即守恒量）。
设 :math:`(M^{2n}, \omega)` 是具有辛形式 :math:`\omega` 的 :math:`2n` 维辛流形，描述具有广义坐标 :math:`q_1, \cdots, q_n` 
和共轭动量 :math:`p_1,\cdots, p_n` 的哈密顿系统，而人们希望找到各种守恒量 :math:`F_1, F_2, \cdots` 来描述系统的演化，
这些守恒量是 :math:`M^{2n}` 上的函数（即相空间点决定了一个守恒量的值）。

**Liouville integrable system 刘维尔可积系统**

:math:`M^{2n}` 上的刘维尔可积系统是指 :math:`M^{2n}` 上 :math:`n` 个函数构成的集合
:math:`\{F_1, F_2, \cdots, F_n\}`，满足下面两个性质：

1. 这些函数的微分 :math:`d F_1, d F_2, \cdots, d F_n` 在 `M^{2n}` 上几乎处处线性无关（“几乎处处”意味着在一个稠密的集合上满足该性质）。

2. 这些函数两两之间的泊松括号为 :math:`0`： :math:`\{F_i, F_j\} = 0, \forall 1\le i, j\le n`。

中心势场运动问题是一个刘维尔可积系统（下面简称可积系统），存在 :math:`H, {|\boldsymbol J|}^2, J_z` 这三个守恒量，满足可积系统的性质。
开普勒运动（二维平面内）是特殊的中心势场运动问题，它有 :math:`H, J_z` 这两个守恒量满足可积系统的性质。可积系统的相空间运动可以
通过积分法完全求解；因此精确求解这样的动力学系统的问题就变成了寻找其独立守恒量的问题。
从上面的例子还可以看到，一个系统的守恒量越多，它的对称性就越高，系统的行为随之呈现出规则性；反之，如果一个系统的守恒量不足，那么它会呈现混沌行为。

