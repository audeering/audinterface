{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}

    .. rubric:: {{ _('Methods') }}

    .. autosummary::
        :nosignatures:
    {% for item in (all_methods)|sort %}
        {%- if not item.startswith('_') or item in ['__call__'] %}
        {{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}

    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
        :nosignatures:
    {% for item in (attributes)|sort %}
        {%- if not item.startswith('_') %}
        {{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}

    .. toctree::
        :hidden:
    {% for item in (all_methods + attributes)|sort %}
        {%- if not item.startswith('_') or item in ['__call__'] %}
        audinterface.{{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}

    {% endblock %}
