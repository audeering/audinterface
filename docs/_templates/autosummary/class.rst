{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}

    .. rubric:: {{ _('Methods, Properties, Attributes') }}

    .. autosummary::
        :toctree:
        :nosignatures:
    {% for item in (all_methods + attributes)|sort(attribute=0) %}
        {%- if not item.startswith('_') or item in ['__call__'] %}
        {{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}

    {% endblock %}
