Specific Model 具体模型
==============================

这一章主要介绍一下 Specific Model 在数学上的定义和目标，以及它在代数化简、排除冗余层面的原理。

微分方程组的极小表示和正则表示
---------------------------------
对于一个微分方程组 :math:`\{a_1, a_2, \cdots, a_m\}`，
其极小表示是指一个极小的子集 :math:`\{a_{i_1}, a_{i_2}, \cdots, a_{i_n}\}`，
满足其他所有微分方程都出现在它所生成的根微分理想中。这里极小的意思是，不能从这个子集中再删除任何一个方程。

这样的极小表示可能有很多个。在我们的程序实现中，
我们可以取其中 `complexity` 字典序最小的一个极小表示。

具体实现方法是，
将 :math:`\{a_1, a_2, \cdots, a_m\}` 按照 complexity 从小到大排序，
然后依次添加到 `diffalg` 中：如果当前添加的方程不属于之前方程所生成的根微分理想，则添加成功，
并将当前方程添加到集合 :math:`S` 中，这样得到的集合被称为微分方程组的正则表示。
正则表示不一定是微分方程组的极小表示，但是我们可以再从后往前依次作一遍检查，
如果某个方程位于除它以外其他方程所生成的根微分理想中，就将它从集合中删去，那么我们最终得到的才是
微分方程组的极小表示。

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