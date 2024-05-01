welcome to AI physicist Documentation
=======================================

this documentation is built by `read the docs`

https://ai-physicist-documentation.readthedocs.io/zh-cn/latest/



how to build the documentation
-------------------------------- 

.. code-block:: console

    git clone git@github.com:catmeow123456/AI-Physicist-documentation.git
    cd AI-Physicist-documentation
    pip install -r requirements.txt
    cd docs
    make html

if you have installed `xelatex` , you can also build the pdf version of the documentation by 
running the following command:

.. code-block:: console

    make latex
    cd build/latex && make all-pdf

or simply type `make latex/pdf` to build the pdf version of the documentation.
