import re
import pandas as pd
import calendar

def get_last_day_of_month(year_month):
    year, month = map(int, year_month.split("-"))
    last_day = calendar.monthrange(year, month)[1]  # Get last day of the month
    return f"{year}-{month:02d}-{last_day:02d}-23-00"

def get_first_time_step(time_input):
    """Returns the first timestamp for a given date input."""
    formats = [
        (r"^\d{4}$", "%Y", "01-01-00-00"),               # YYYY → YYYY-01-01-00-00
        (r"^\d{4}-\d{2}$", "%Y-%m", "01-00-00"),         # YYYY-mm → YYYY-mm-01-00-00
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d", "00-00")    # YYYY-mm-dd → YYYY-mm-dd-00-00
    ]

    for pattern, fmt, suffix in formats:
        if re.match(pattern, time_input):
            dt = datetime.strptime(time_input, fmt)
            return dt.strftime(f"%Y-%m-%d-{suffix}")

    raise ValueError("Invalid date format")

def get_last_time_step(time_input):
    """Returns the last timestamp for a given date input."""
    formats = [
        (r"^\d{4}$", "%Y", "%Y-12-31-23-59"),          # YYYY → YYYY-12-31-23-59
        (r"^\d{4}-\d{2}$", "%Y-%m", "%Y-%m-last"),     # YYYY-mm → YYYY-mm-last_day-23-59
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d", "%Y-%m-%d-23-59")  # YYYY-mm-dd → YYYY-mm-dd-23-59
    ]

    for pattern, fmt, output_fmt in formats:
        if re.match(pattern, time_input):
            dt = datetime.strptime(time_input, fmt)
            
            if "last" in output_fmt:  # If we need the last day of the month
                next_month = dt.replace(day=28) + timedelta(days=4)  # Ensure we're in next month
                last_day = (next_month - timedelta(days=next_month.day)).day
                return dt.strftime(f"%Y-%m-{last_day}-23-59")
            
            return dt.strftime(output_fmt)

    raise ValueError("Invalid date format")