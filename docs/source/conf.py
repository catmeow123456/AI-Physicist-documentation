# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'AI-Physicist'
copyright = '2024, cat123456'
author = 'cat123456'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


# -- for latex

latex_engine = 'xelatex'
latex_elements = {
    'preamble': r'''
    \usepackage[UTF8, scheme = plain]{ctex}
    '''
}
# latex_elements = {
#     'fontpkg': r'''
# \setmainfont{DejaVu Serif}
# \setsansfont{DejaVu Sans}
# \setmonofont{DejaVu Sans Mono}
# ''',
#     'preamble': r'''
# \usepackage[titles]{tocloft}
# \cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
# \setlength{\cftchapnumwidth}{0.75cm}
# \setlength{\cftsecindent}{\cftchapnumwidth}
# \setlength{\cftsecnumwidth}{1.25cm}
# ''',
#     'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
#     'printindex': r'\footnotesize\raggedright\printindex',
# }
latex_show_urls = 'footnote'
