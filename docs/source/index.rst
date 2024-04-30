Welcome to AI-Physicist's documentation!
========================================

人类通过研究各种物理问题，从简单问题中获得知识和经验，逐步解决越来越复杂的问题。AI 亦然。
给AI一些包含若干实验在内的具有有限精度的实验数据，我们期望它能够通过综合考察这些实验，合理安排考察不同实验的顺序，最终能对每个实验都构建出物理可解释的模型。
这样做的意义有两方面，其一是帮助人类科学家发现新的物理规律，其二是为已有的物理学理论提供新的理论形式（正如哈密顿力学和拉格朗日力学的提出之于经典力学理论体系一样），
其三是启发通用人工智能（AGI）的发展（它必须像人类一样具备从环境当中学习知识的能力）。

**AI-Physicist** is a Python library for automatically discovering physical laws from experimental data.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.


.. toctree::
   introduction
   reference

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Contents

   usage
   api

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Explanation

   structure/index
   language/index
   knowledge/index
   symbolic/index
   aiphysicist/index

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: MileStone

   milestone/index
