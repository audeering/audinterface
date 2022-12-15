{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}
    {%- for item in (all_methods + attributes)|sort %}
        {%- if not item.startswith('_') or item in ['__call__'] %}
{{ item | escape | underline(line='-') }}
            {%- if item in all_methods %}
.. automethod:: {{ name }}.{{ item }}
            {%- elif item in attributes %}
.. autoattribute:: {{ name }}.{{ item }}
            {%- endif %}
        {% endif %}
    {%- endfor %}
    {% endblock %}
