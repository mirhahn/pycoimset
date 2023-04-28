{{ fullname | escape | underline }}

Description
-----------

.. automodule:: {{ fullname | escape }}
   

{% if modules %}
Modules
-------
.. autosummary::
   :toctree:
   :recursive:

   {% for module in modules -%}
       {{ module }}
   {% endfor %}
{% endif %}

{% if classes %}
Classes
-------
.. autosummary::
   :toctree:
   :recursive:

   {% for class in classes -%}
       {{ class }}
   {% endfor %}
{% endif %}

{% if functions %}
Functions
---------
.. autosummary::

   {% for function in functions -%}
       {{ function }}
   {% endfor %}
{% endif %}

