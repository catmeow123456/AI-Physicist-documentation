Differential Algebra 微分代数
==============================

参考 [rosenfeld]_。
如无特殊说明符号 :math:`\mathbb{K}` 代表特征为 :math:`0` 的域
（例如有理数域 :math:`\mathbb{Q}` ，实数域 :math:`\mathbb{R}` ）。

glossary 术语表
---------------------
**differential ring 微分环**

在数学上，一个环 :math:`R` 上如果再配有有限多个导子（derivation）
:math:`\partial_1, \partial_2, \ldots, \partial_m`，

这些导子是 :math:`R` 上的环同态，它们两两对易（commute）
:math:`\partial_i \partial_j r = \partial_j \partial_i r, \forall r \in R`，

且满足莱布尼茨法则
:math:`\partial_i(ab) = a\partial_i(b) + b\partial_i(a)`，

则将 :math:`\langle R, \delta_i(1\le i\le m)\rangle` 称为一个微分环。

**differential polynomial ring 微分多项式环**

对于给定的导子 :math:`\partial_1, \partial_2, \ldots, \partial_m`，
微分算子（derivation operator）是指具有形式 :math:`\theta = \partial_1^{a_1}\ldots \partial_n^{a_m}` 的算子，

给定微分变元（differential indeterminate） :math:`f_1, f_2, \ldots, f_n` （也就是物理学家所处理的函数），
可以在微分变元前添加微分算子，得到的 :math:`\theta f`
项被称作是 :math:`f` 的导数（derivatives）。
这些微分变元构成了一个多项式环 :math:`R = \mathbb{K}[f_1, f_2, \ldots, f_n]`，而我们要研究的是带有导子的
微分多项式环 :math:`\mathrm{diffR} = \langle R, \partial_i(1\le i\le m)\rangle`。

将所有可能的 :math:`\theta f` 所构成的集合记为 :math:`\Theta F`。
那么微分多项式环实际上可以看作是由 :math:`\Theta F` 中所有元素所生成的普通多项式环 :math:`\mathrm{diffR} = \mathbb{K}[\Theta F]`。

在物理上可以限制 :math:`f` 是关于哪些变量（variable）的函数，从而它前面只能有和变量相对应的导子。
例如 :math:`f=f(t)`，那么 :math:`f` 前面的微分算子只能是 :math:`\partial_t^{a_t}, a_t \in \mathbb{N}`。
特别地，物理学常数（例如光速 :math:`c`，引力常数 :math:`G`，库伦常数 :math:`k`） 可以看作是不依赖于任何变量的函数 :math:`f=f()`。

例子：麦克斯韦方程组

.. math::

    &\nabla \cdot \boldsymbol E(x, y, z, t) = \frac{1}{\epsilon_0} \rho(x, y, z, t)\\
    &\nabla \times \boldsymbol E(x, y, z, t) = -\partial_t \boldsymbol B(x, y, z, t)\\
    &\nabla \cdot \boldsymbol B(x, y, z, t) = 0\\
    &\nabla \times \boldsymbol B(x, y, z, t) = \mu_0 \boldsymbol J(x, y, z, t) + \mu_0 \epsilon_0 \partial_t \boldsymbol E(x, y, z, t)
  
这四条方程都是微分多项式环 :math:`\langle R=\mathbb{K}[\epsilon_0, \mu_0, E_x,E_y,E_z,B_x,B_y,B_z,J_x,J_y,J_z,\rho], \partial_x, \partial_y, \partial_z, \partial_t\rangle`
中的元素。

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

**differential ideal 微分理想**

对于微分多项式环 :math:`\mathrm{diffR} = \mathbb{K}[\Theta F]` 上的一个理想 :math:`I`，如果它对求导运算封闭，
那么称 :math:`I` 是一个微分理想。

取 :math:`\mathrm{diffR}` 的一个子集 :math:`A\subset \mathrm{diffR}`，我们记 :math:`[A]` 为由 :math:`A` 生成的微分理想，
它是在 :math:`\mathrm{diffR}` 中包含 :math:`A` 的最小的微分理想。

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

**differential radical ideal 微分根理想**

对于 :math:`\mathrm{diffR}` 上的微分理想 :math:`I`，
它的根理想 :math:`\sqrt{I}` 被定义为
:math:`\{p\in \mathrm{diffR} | \exists n \in \mathbb{N}, p^n \in I\}`。

**differential prime ideal 微分素理想**

对于 :math:`\mathrm{diffR}` 上的微分理想 :math:`I`，
如果对于任意的 :math:`p, q \in \mathrm{diffR}`，
当 :math:`pq \in I` 时，至少有一个 :math:`p \in I` 或者 :math:`q \in I`，
那么称 :math:`I` 是一个微分素理想。

**rosenfeld groebner 算法**

基于两个重要的定理：

Any radical differential ideal :math:`\mathfrak{r}` 
of a differential polynomial ring :math:`\mathrm{R}`
is a finite intersection of differential prime ideals which is unique when minimal.
([Chap. III, Sect. 4, Theorem 1] [diffalgbook]_)

The following is a differential analog of Hilbert's theorem of zeros ([Chap. IV, Sect. 2] [diffalgbook]_):

Theorem 2 (theorem of zeros) 
Let :math:`R = \mathbb{K}[U]` be a differential polynomial ring over a differential field
of characteristic zero and :math:`\mathfrak{r}` be a differential ideal of :math:`R`.
A differential polynomial :math:`p` 
vanishes on every solution of :math:`\mathfrak{r}`, in any differential field extension of :math:`\mathbb{K}`,
if and only if :math:`p \in \sqrt{\mathfrak{r}}`.

rosenfeld groebner 算法实现了将微分理想 :math:`I` 表达为有限个微分素理想的交集 :math:`I_1 \cap I_2 \ldots \cap I_k`，
每个微分素理想都被它的一组 differential groebner basis 表达，
它被称作是正则微分链（regular differential chain），
利用正则微分链，可以很方便地判断一条给定的微分方程是否出现在这个微分素理想 :math:`I_i` 当中。
于是，对于微分方程 :math:`\mathrm{eq} = 0`，当且仅当 :math:`\forall i, \mathrm{eq} \in I_i`，
:math:`\mathrm{eq}` 出现在微分理想 :math:`I` 当中。

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