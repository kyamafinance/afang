from datetime import datetime


def milliseconds_to_datetime(milliseconds: int) -> datetime:
    """Convert a UNIX timestamp in milliseconds to a datetime object.

    :param milliseconds: UNIX timestamp in milliseconds.

    :return: datetime
    """

    return datetime.utcfromtimestamp(milliseconds / 1000)
