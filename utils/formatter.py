import datetime


def get_formatted_time(start, end):
    """Formats a time difference between two datetime objects into a human-readable string.

      Args:
        start: A datetime object representing the start time.
        end: A datetime object representing the end time.

      Returns:
        A string representing the time difference in the format "Xh Ym Zs", 
        where X, Y, and Z are the number of hours, minutes, and seconds, respectively.
        If the time difference is less than 1 second, it returns "0s".
      """
    diff = end - start
    time_obj = datetime.timedelta(seconds=diff)
    diff = int(time_obj.total_seconds())
    parts = []
    if diff >= 3600:
        h, diff = divmod(diff, 3600)
        parts.append(f'{h}h')
    if diff >= 60:
        m, diff = divmod(diff, 60)
        parts.append(f'{m}m')
    if diff > 0 or not parts:
        parts.append(f'{diff}s')
    return ' '.join(parts)
