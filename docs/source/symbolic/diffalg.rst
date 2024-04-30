Differential Algebra 微分代数
==============================

参考 [rosenfeld]_、[rosenfeldbook]_、[diffalgbook]_、[computediffalg]_。

Theory of differential algebra 微分代数理论
-------------------------------------------------

basic notions of differential algebra 微分代数的基本概念
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**differential ring 微分环**

一个环 :math:`R` 上如果再配有有限多个导子（derivation）
:math:`\partial_1, \partial_2, \ldots, \partial_m`，

这些导子是 :math:`R` 到 :math:`R` 的映射，它们两两对易（commute）
:math:`\partial_i \partial_j r = \partial_j \partial_i r, \forall r \in R`，

且满足线性性
:math:`\forall a,b \in R, \partial_i(a+b) = \partial_i(a) + \partial_i(b)`，
和莱布尼茨法则
:math:`\forall a,b \in R, \partial_i(ab) = a\partial_i(b) + b\partial_i(a)`，

则将 :math:`R = \langle R, \partial_i(1\le i\le m)\rangle` （下面简记为 :math:`R` ）称为一个微分环。

**differential ideal 微分理想**

对于微分环 :math:`R` 上的一个理想 :math:`I`，如果它对求导运算封闭，
那么称 :math:`I` 是一个微分理想。

取 :math:`R` 的一个子集 :math:`A\subset R`，我们记 :math:`[A]` 为由 :math:`A` 生成的微分理想，
它是在 :math:`R` 中包含 :math:`A` 的最小的微分理想。

**radical differential ideal 根微分理想**

对于微分环 :math:`R` 上的微分理想 :math:`I`，
它的根理想 :math:`\sqrt{I}` 被定义为
:math:`\{p\in R | \exists n \in \mathbb{N}, p^n \in I\}`。
如果一个微分理想 :math:`I` 的根理想 :math:`\sqrt{I}` 等于它自身，那么它是一个根微分理想。

**differential prime ideal 微分素理想**

对于微分环 :math:`R` 上的微分理想 :math:`I`，
如果对于任意的 :math:`p, q \in R`，
当 :math:`pq \in I` 时，至少有一个 :math:`p \in I` 或者 :math:`q \in I`，
那么称 :math:`I` 是一个微分素理想。

**decomposition of radical differential ideals 根微分理想的分解**

在实际应用中，我们经常需要将一个根微分理想分解为若干个微分素理想的交集，因为微分素理想是更容易处理的对象。
以下定理表明了在数学上，这样的分解总是可能的。

    *定理 1*. [Chap. III, Sect. 4, Theorem 1] [diffalgbook]_

    :math:`R` 是一个微分环，对于其中任意的根微分理想 :math:`\mathfrak{r} \subsetneq R`
    （ :math:`\mathfrak{r} = \sqrt{\mathfrak{r}}`），它一定可以被分解为
    有限个微分素理想的交集。

differential polynomial ring 微分多项式环
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在物理学中，我们经常需要解决一些微分方程，这些微分方程可以被看作是微分多项式环中的元素。
例如在自由落体实验中，设小球的高度关于时间的函数为 :math:`z(t)`，则有方程组
:math:`v(t) - g \cdot t - v_0 = 0,\ v(t) - \partial_t z(t) = 0`。

其中变量（variable） :math:`t` 和导子 :math:`\partial_t` 有对应关系，
在这个问题中，可以将 :math:`\mathbb{K} = \mathbb{Q}(t)` （也就是关于 :math:`t` 的有理函数域）
作为我们关心的数学结构，因为 :math:`\partial_t: \mathbb{Q}(t)\to \mathbb{Q}(t)` 是其上的一个导子，所以它又被叫做微分域。
:math:`\mathbb{Q}` 也是一个特殊的微分域，因为所有的导子都将 :math:`\mathbb{Q}` 中的元素映射为 :math:`0`。

而 :math:`z(t),v(t)` 和 :math:`g,v_0` 则是微分变元，而上述自由落体方程组可以看作微分多项式环
:math:`\mathbb{K}[g,v_0,z(t),v(t), \partial_t z(t), \partial_t v(t), \cdots]` 中的一个元素。

下面我们给出微分域与微分多项式环的具体定义：

**differential field 微分域**

一个域 :math:`\mathbb{K}` 上如果再配有有限多个导子（derivation）
:math:`\partial_{1}, \partial_{2}, \ldots, \partial_{m}`，

这些导子是 :math:`\mathbb{K}` 到 :math:`\mathbb{K}` 的映射，它们两两对易（commute）
:math:`\partial_{i} \partial_{j} r = \partial_{j} \partial_{i} r, \forall r \in \mathbb{K}`，

且满足线性性
:math:`\forall a,b \in \mathbb{K}, \partial_i(a+b) = \partial_i(a) + \partial_i(b)`，
和莱布尼茨法则
:math:`\forall a,b \in \mathbb{K}, \partial_i(ab) = a\partial_i(b) + b\partial_i(a)`，

则将 :math:`\langle \mathbb{K}, \partial_{i}(1\le i\le m)\rangle` （下面直接简写为 :math:`\mathbb{K}` ） 称为一个微分域。

**differential polynomial ring 微分多项式环**

对于给定的导子 :math:`\partial_1, \partial_2, \ldots, \partial_m` 及相应的微分域 :math:`\mathbb{K}`，
微分算子（derivation operator）是指具有形式 :math:`\theta = \partial_1^{a_1}\ldots \partial_n^{a_m}` 的算子，
特别地，恒等映射 `\theta = \mathrm{id}` 是平凡的微分算子。

给定微分变元（differential indeterminate） :math:`f_1, f_2, \ldots, f_n` （也就是物理学家所处理的函数），
可以在微分变元前添加微分算子，得到的 :math:`\theta f` 项被称作是 :math:`f` 的导数（derivatives）。
将所有可能的 :math:`\theta f` 所构成的集合记为 :math:`\Theta F`。
那么这些微分变元及它们的导数构成了一个多项式环 :math:`R = \mathbb{K}[\Theta F]`，
不难验证可以在 :math:`R` 上定义导子 :math:`\partial_1, \partial_2, \ldots, \partial_m`，使它们互相对易、满足线性性和莱布尼兹法则。
因此 :math:`R = \langle R, \partial_i(1\le i\le m)\rangle` （下面简记为 :math:`R` ）被称作是一个微分多项式环。

在物理上可以限制 :math:`f` 是关于哪些变量的函数，从而它前面只能有和变量相对应的导子。
例如 :math:`f=f(t)`，那么 :math:`f` 前面的微分算子只能是 :math:`\partial_t^{a_t}, a_t \in \mathbb{N}`。
特别地，物理学常数（例如光速 :math:`c`，引力常数 :math:`G`，库伦常数 :math:`k`） 可以看作是不依赖于任何变量的函数 :math:`f=f()`。

例子：麦克斯韦方程组

.. math::

    &\nabla \cdot \boldsymbol E(x, y, z, t) = \frac{1}{\epsilon_0} \rho(x, y, z, t)\\
    &\nabla \times \boldsymbol E(x, y, z, t) = -\partial_t \boldsymbol B(x, y, z, t)\\
    &\nabla \cdot \boldsymbol B(x, y, z, t) = 0\\
    &\nabla \times \boldsymbol B(x, y, z, t) = \mu_0 \boldsymbol J(x, y, z, t) + \mu_0 \epsilon_0 \partial_t \boldsymbol E(x, y, z, t)
  
这四条方程都是微分多项式环 :math:`R = \langle R=\mathbb{K}[\epsilon_0, \mu_0, E_x,E_y,E_z,B_x,B_y,B_z,J_x,J_y,J_z,\rho], \partial_x, \partial_y, \partial_z, \partial_t\rangle`
中的元素，这里的微分域 :math:`\mathbb{K}` 可以取作 :math:`\mathbb{Q},\mathbb{R}` 或 :math:`\mathbb{C}`，这足以表达麦克斯韦方程组。
也可以取作 :math:`\mathbb{Q}(x,y,z,t)` 即关于 :math:`x,y,z,t` 的有理函数域，
此时微分多项式环 :math:`R` 能表达的微分方程更加广泛。

.. code-block:: python
    :emphasize-lines: 16,17,18

    import sympy as sp
    from aiphysicist.diffalg import DifferentialRing
    t, x, y, z = sp.symbols('t x y z')
    Ex = sp.Function('Ex')(x,y,z,t)
    Ey = sp.Function('Ey')(x,y,z,t)
    Ez = sp.Function('Ez')(x,y,z,t)
    Bx = sp.Function('Bx')(x,y,z,t)
    By = sp.Function('By')(x,y,z,t)
    Bz = sp.Function('Bz')(x,y,z,t)
    Jx = sp.Function('Jx')(x,y,z,t)
    Jy = sp.Function('Jy')(x,y,z,t)
    Jz = sp.Function('Jz')(x,y,z,t)
    rho = sp.Function('rho')(x,y,z,t)
    eps0 = sp.Symbol('eps0')
    mu0 = sp.Symbol('mu0')
    diffR = DifferentialRing.default(
        [eps0, mu0, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho]
    )

下面是一个在一个微分多项式环上定义微分理想的例子：

.. code-block:: python
    :emphasize-lines: 9,10

    import sympy as sp
    from aiphysicist.diffalg import DifferentialRing, diffalg
    t, mass, x0, v0, a0 = sp.symbols('t mass x0 v0 a0')
    x = sp.Function('x')(t)
    xr = sp.Function('xr')(t)
    diffring = DifferentialRing.default([x, xr, mass, x0, v0, a0])
    v = sp.Derivative(x, t)
    a = sp.Derivative(v, t)
    eqs = [a - a0, v - a*t  - v0, x - a*t**2/2 - v0*t - x0]
    ideal = diffalg.from_eqs(diffring, eqs)

>>> print(ideal)
DifferentialAlgebra:
[-a0*t**2 - 2*t*v0 + 2*x - 2*x0]
>>> eq = x - v*t/2 - x0 - v0*t/2
>>> print(ideal.belongs_to(eq))
True

**Solutions of differential equations 微分方程的解**

在物理学中，我们经常关心微分方程的解，希望知道它与实验观测结果在误差范围内是否一致。
许多微分方程具有形式幂级数（formal power series）解，或是洛朗级数（Laurent series）解。
因此当我们讨论微分方程的解时，是在一个更大的微分域上讨论的。

例如在方程 :math:`\partial_t x(t) + c\cdot x(t) = 0` 中，
:math:`x(t)` 具有幂级数解 :math:`x(t) \propto \sum_{n=0}^{\infty} \frac{1}{n!} c^n t^n`，
因此可以设置微分变元的解的范围 :math:`c\in \mathbb{R}, x(t) \in \mathbb{R}[[t]]` ，在这个范围中讨论这个微分方程的解。
:math:`\mathbb{R}[[t - a]]` 代表关于在 :math:`a\in \mathbb{R}` 处的形式幂级数所构成的域。
或者也可以在 :math:`c\in \mathbb{C}, x(t) \in \mathbb{C}((t))` 的范围内讨论微分方程的解。
:math:`\mathbb{C}((t - z))` 代表关于在 :math:`z\in \mathbb{C}` 处的形式 Laurent 展开所构成的域。

对于微分多项式环 :math:`R=\mathbb{K}[U]` 中的一个微分理想 :math:`\mathfrak{r}`，我们定义
:math:`\mathfrak{r}` 在微分域 :math:`\mathbb{K'}` 上的解（solution）为：
:math:`\mathbb{K'}` 上满足 :math:`\mathfrak{r}` 中所有方程的解（“满足”的意思是，将解代入方程后，可以得到 :math:`0`）
所构成的集合。
这里 :math:`\mathbb{K'}` 是 :math:`\mathbb{K}` 的一个微分域扩张（differential field extension），
当我们在不同的扩域上讨论时，会得到不同的解的集合，甚至不同的解的形式。

下面这条定理将微分方程的解的讨论与微分理想联系了起来，它类似于希尔伯特零点定理，只不过它是在微分代数中讨论的：

    *定理 2*. 零点定理 [Chap. IV, Sect. 2] [diffalgbook]_

    :math:`\mathrm{R}=\mathbb{K}[U]` 是一个微分多项式环， :math:`\mathbb{K}` 为特征 :math:`0` 的微分域。
    :math:`\mathfrak{r}` 是 :math:`\mathrm{R}` 的一个微分理想。
    那么微分多项式 :math:`p \in \sqrt{\mathfrak{r}}` 当且仅当
    对于任意 :math:`\mathbb{K}` 的微分域扩张 :math:`\mathbb{K}'`，
    :math:`\mathfrak{r}` 在 :math:`\mathbb{K}'` 上的解总是满足 :math:`p`。

因此当我们想要知道微分方程组 :math:`p_1,\cdots, p_n` 的解是否一定满足微分方程 :math:`p_{n+1}` 时，只需要
求出微分方程组所生成的根微分理想 :math:`\sqrt{[p_1,\cdots, p_n]}`，然后检查 :math:`p_{n+1}` 是否属于这个根理想即可。

**Rosenfeld Groebner 算法**

[rosenfeldbook]_

Rosenfeld Groebner 算法实现了将微分理想 :math:`I` 的根理想 :math:`\sqrt{I}` 
分解为有限个正则微分理想 （regular differential ideal） 的交集，这一步被称为正则分解（regular decomposition）：
（正则微分理想的定义： TODO）

.. math::

    \sqrt{I} = I_1 \cap I_2 \ldots \cap I_k

分解得到的结果可能是冗余（redundant）的，即等式右边的某个 :math:`I_i` 有可能可以删掉。
如何判断一个正则微分理想在一个分解中是否冗余，是微分代数中的一个著名的开放问题。尽管 Rosenfeld Groebner 算法得到的分解
结果可能是冗余的，但它不妨碍我们利用这个分解来判断某一方程是否属于其根理想。

进一步地，可以利用准素分解（primary decomposition）算法对每个正则微分理想作进一步的分解，
我们可以得到 :math:`\sqrt{I}` 的一个微分素理想分解（这一分解仍然可能是冗余的）。

Rosenfeld Groebner 算法最终将每个正则微分理想用它的约化（reduced）groebner basis 表达，
最终返回一个正则微分链 （regular differential chain） 的列表。

因此根据 groebner basis 的性质，可以很方便地判断一条给定的微分方程是否出现这个微分理想 :math:`I_i` 当中。
于是，对于微分方程 :math:`\mathrm{eq} = 0`，:math:`\mathrm{eq}` 属于根微分理想 :math:`\sqrt{I}`，
当且仅当 :math:`\forall i, \mathrm{eq} \in I_i`。

下面是使用 Rosenfeld Groebner 算法的一个例子：

.. code-block:: python
    :emphasize-lines: 16,17

    import sympy as sp
    from aiphysicist.diffalg import DifferentialRing, diffalg
    t, mass1, mass2, P, E = sp.symbols('t mass1 mass2 P0 E0')
    x1 = sp.Function('x1')(t)
    x2 = sp.Function('x2')(t)
    ring = DifferentialRing([('lex', [x1, x2]),
                             ('lex', [P, E]),
                             ('lex', [mass1, mass2])])
    x1, x2 = sp.symbols('x1 x2')
    v1 = sp.Derivative(x1, t)
    v2 = sp.Derivative(x2, t)
    p1 = mass1 * v1
    p2 = mass2 * v2
    e1 = mass1 * v1**2 / 2
    e2 = mass2 * v2**2 / 2
    eqs = [p1 + p2 - P, e1 + e2 - E]
    ideal = diffalg.from_eqs(ring, eqs)

>>> print(ideal)
DifferentialAlgebra:
[-P0 + mass1*Derivative(x1, t) + mass2*Derivative(x2, t), -2*E0*mass1 + P0**2 - 2*P0*mass2*Derivative(x2, t) + mass1*mass2*Derivative(x2, t)**2 + mass2**2*Derivative(x2, t)**2]
[-P0 + mass1*Derivative(x1, t) + mass2*Derivative(x1, t), -P0 + mass1*Derivative(x2, t) + mass2*Derivative(x2, t), -2*E0*mass1 - 2*E0*mass2 + P0**2]
[Derivative(x1, t), Derivative(x2, t), P0, E0]
[-2*E0*mass2 + P0**2 + 2*P0*mass2*Derivative(x1, t), -2*E0*mass2 - P0**2 + 2*P0*mass2*Derivative(x2, t), mass1 + mass2]
[Derivative(x1, t) - Derivative(x2, t), P0, E0, mass1 + mass2]
[-P0 + mass1*Derivative(x1, t), -2*E0*mass1 + P0**2, mass2]
[Derivative(x1, t), P0, E0, mass2]
[-P0 + mass2*Derivative(x2, t), -2*E0*mass2 + P0**2, mass1]
[Derivative(x2, t), P0, E0, mass1]
[P0, E0, mass1, mass2]
>>> print(ideal.gb[0].reduce((v1-v2)**2 * mass1 * mass2))
2*E0*mass1 + 2*E0*mass2 - P0**2
>>> print(ideal.belongs_to(P**2 - 2*E*(mass1 + mass2) + (v1-v2)**2 * mass1 * mass2))
True