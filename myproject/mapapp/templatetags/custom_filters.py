# templatetags/custom_filters.py
from django import template
from datetime import datetime
register = template.Library()


@register.filter
def sum_attribute(value, arg):
    return sum(item[arg] for item in value if arg in item and item[arg] is not None)


@register.filter
def average_attribute(value, arg):
    values = [item[arg] for item in value if arg in item and item[arg] is not None]
    return sum(values) / len(values) if values else 0


@register.filter(name='is_datetime')
def is_datetime(value):
    return isinstance(value, datetime)


@register.filter
def parse_date(value):
    if isinstance(value, str):
        try:
            # Attempt to parse the date from ISO format
            return datetime.fromisoformat(value).strftime("%Y-%m-%d")
        except ValueError:
            # If it's not a date, return the value as is
            return value
    else:
        # If the value is not a string, return it as is
        return value
