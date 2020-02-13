from datetime import datetime
import numpy as np
import pandas as pd
import os
from PyPDF2 import PdfFileMerger, PdfFileReader
from time import sleep


def homedir():
    """Return file path for home directory (Windows or Linux)"""
    return os.path.expanduser('~')


def makelist(input):
    """Return a list/array object from the input.

    If input is a single value (e.g. int, str, etc.) then output
    is a list of length 1.  If input is already a list or array, then
    output is the same as input.
    """
    if isinstance(input, list) or isinstance(input, np.array):
        output = input
    else:
        output = [input]
    return output


def disptime(fmt='%Y-%m-%d %H:%M:%S'):
    now = datetime.now().strftime(fmt)
    print(now)
    return now


def homedir(options=['/home/jennifer/', '/home/jwalker/',
            'C:/Users/jenfl/']):
    """Return home directory for this computer."""

    home = None
    for h in options:
        if os.path.isdir(h):
            home = h
    if home is None:
        raise ValueError('Home directory not found in list of options.')
    return home


def pdfmerge(filenames, outfile, delete_indiv=False, wait=0.5):
    """Merge PDF files into a single file."""

    # Merge the files
    merger = PdfFileMerger()
    for filename in filenames:
        merger.append(PdfFileReader(open(filename, 'rb')))
    merger.write(outfile)

    # Delete the individual files
    if delete_indiv:
        for filename in filenames:
            # Sleep a moment to try to avoid Windows weirdness with open files
            if wait is not None:
                sleep(wait)
            os.remove(filename)

    return None


# =============================================================================
# Calendar utility functions
# =============================================================================

def isleap(year):
    """Return True if year is a leap year, False otherwise."""
    return year % 4 == 0


def month_str(month, upper=True):
    """Returns the string e.g. 'JAN' corresponding to month"""

    months=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
            'sep', 'oct', 'nov', 'dec']

    mstr = months[month - 1]
    if upper:
        mstr = mstr.upper()
    return mstr


def days_per_month(leap=False):
    """Return array with number of days per month."""

    ndays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leap:
        ndays[1]+= 1
    return ndays


def days_this_month(year, month):
    """Return the number of days in a selected month and year.

    Both inputs must be integers, and month is the numeric month 1-12.
    """
    ndays = days_per_month(isleap(year))
    return ndays[month - 1]


def season_months(season):
    """
    Return list of months (1-12) for the selected season.

    Valid input seasons are:
    ssn=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
         'sep', 'oct', 'nov', 'dec', 'djf', 'mam', 'jja', 'son',
         'mayjun', 'julaug', 'marapr', 'jjas', 'ond', 'ann']
    """

    ssn=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
         'sep', 'oct', 'nov', 'dec', 'djf', 'mam', 'jja', 'son',
         'mayjun', 'julaug', 'marapr', 'jjas', 'ond', 'ann']

    imon = [1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, [1,2,12], [3,4,5], [6,7,8], [9,10,11],
            [5,6], [7,8], [3,4], [6,7,8,9], [10,11,12], list(range(1,13))]

    try:
        ifind = ssn.index(season.lower())
    except ValueError:
        raise ValueError('Season not found! Valid seasons: ' + ', '.join(ssn))

    months = imon[ifind]

    # Make sure the output is a list
    if isinstance(months, int):
        months =[months]

    return months


def season_days(season, leap=False):
    """
    Returns indices (1-365 or 1-366) of days of the year for the input season.

    Valid input seasons are as defined in the function season_months().
    """

    # Index of first day of each month
    ndays = days_per_month(leap=leap)
    ndays.insert(0,1)
    days = np.cumsum(ndays)

    # Index of months for this season
    imon = season_months(season)

    # Days of the year for this season
    if isinstance(imon, list):
        # Iterate over months in this season
        idays=[]
        for m in imon:
            idays.extend(list(range(days[m-1], days[m])))
    else:
        # Single month
        idays = list(range(days[imon-1], days[imon]))

    return idays


def jday_to_mmdd(jday, year=None):
    """
    Returns numeric month and day for day of year (1-365 or 1-366).

    If year is None, a non-leap year is assumed.
    Usage: mon, day = jday_to_mmdd(jday, year)
    """
    if year is None or not isleap(year):
        leap = False
    else:
        leap = True

    ndays = days_per_month(leap)
    iday = np.cumsum(np.array([1] + ndays))
    if jday >= iday[-1]:
        raise ValueError('Invalid input day %d' + str(jday))

    BIG = 1000 # Arbitrary big number above 366
    d = np.where(jday >= iday, jday - iday + 1, BIG)
    ind = d.argmin()
    mon = ind + 1
    day = d[ind]

    return mon, day


def mmdd_to_jday(month, day, year=None):
    """
    Returns Julian day of year (1-365 or 1-366) for day of month.

    If year is None, a non-leap year is assumed.
    Usage: mon, day = jday_to_mmdd(jday, year)
    """
    if year is None or not isleap(year):
        leap = False
    else:
        leap = True

    days = season_days('ann', leap)
    mmdd = {}
    for mm in range(1, 13):
        mmdd[mm] = {}
    for d in days:
        mm, dd = jday_to_mmdd(d, year)
        mmdd[mm][dd] = d
    jday = mmdd[month][day]

    return jday
