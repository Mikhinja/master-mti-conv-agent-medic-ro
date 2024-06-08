
from datetime import timedelta


def timedelta_str(dt:timedelta)->str:
    s = dt.seconds
    return f'{s//3600:>2}:{(s//60)%60:>2}:{s%60:>2}s'
