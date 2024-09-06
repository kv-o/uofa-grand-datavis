#!/usr/bin/python3

import matplotlib.pyplot as pyplot
import numpy
import pandas

# for cleaning the records_lost column later

def intify(x):
    return int(x)

def clean(x):
    x = x.replace(",", "")
    return int(x)

# password breach dataset:
# https://www.kaggle.com/datasets/joebeachcapital/worlds-biggest-data-breaches-and-hacks?resource=download
breaches = pandas.read_csv("kaggle.csv")

# drop the first row and the "unnamed" column
breaches = breaches.iloc[1:]
breaches = breaches.drop(columns=["Unnamed: 11"])

# rename the columns to something workable
breaches = breaches.rename(columns={
    "organisation": "host",
    "alternative name": "alt_name",
    "records lost": "compromises",
    "year   ": "year",
    "date": "date",
    "story": "story",
    "sector": "sector",
    "method": "method",
    "interesting story": "interesting_story",
    "data sensitivity": "data_sensitivity",
    "displayed records": "displayed_records",
    "source name": "src_name",
    "1st source link": "src_link1",
    "2nd source link": "src_link2",
    "ID": "id"
})

# apply clean function to compromises column
# to convert string values to integer values
breaches["compromises"] = breaches["compromises"].apply(clean)

# drop unneeded columns in kaggle dataset, then
# add auth_method column with all fields in kaggle dataset set to "password"
breaches = breaches.drop(columns=["alt_name", "story", "sector", "method", "interesting_story", "data_sensitivity", "displayed_records", "src_name", "id"])
breaches = breaches.assign(auth_method=pandas.Series(["password"] * len(breaches.index)))
breaches["year"] = breaches["year"].apply(intify)

# read manually collated dataset
# https://github.com/kv-o/uofa-grand-datavis/blob/main/collated.csv
data = pandas.read_csv("collated.csv")
data["year"] = data["year"].apply(intify)

# concatenate two datasets
data = pandas.concat([data, breaches])

# get a list of years for which data is available
# store password compromise data from 2004 because it is elided later for unknown reasons
sums = {}
years = []
compromises04 = 0
hosts04 = len(data[data.year == 2004].host)
for year in data.year.unique():
    years.append(int(year))
    for n in data[data.year == year].compromises:
        if int(year) == 2004:
            compromises04 += n

# make sure list of years has all values from minyear to maxyear
minyear = min(years)
maxyear = max(years)
years = []
while minyear != maxyear + 1:
    years.append(int(minyear))
    minyear += 1

# add zero placeholder values so matplotlib does not complain about mismatched data sizes
for method in data.auth_method.unique():
    nilyears = list(set(years).difference(data[data.auth_method == method].year.unique()))
    for year in nilyears:
        data.loc[-1] = ["", 0, year, "", method, "", ""]
        data.index = data.index + 1
        data = data.sort_index()

# group all key compromises count by year
mfa = data[data.auth_method == "2fa"].groupby("year").agg(
    {"host": "count", "compromises": "sum"}
).reset_index()

sms = data[data.auth_method == "sms"].groupby("year").agg(
    {"host": "count", "compromises": "sum"}
).reset_index()

ssh = data[data.auth_method == "ssh"].groupby("year").agg(
    {"host": "count", "compromises": "sum"}
).reset_index()

passwords = data[data.auth_method == "password"].groupby("year").agg(
    {"host": "count", "compromises": "sum"}
).reset_index()

# add missing passwords entry for 2004 back to the passwords dataset
passwords = passwords[passwords.compromises != 0]
passwords.loc[-1] = [2004, hosts04, compromises04]
passwords.index = passwords.index + 1
passwords = passwords.sort_index()
passwords = passwords.sort_values(by=["year"], ascending=True)

# get figure and axes objects from pyplot
fig, axes = pyplot.subplots()
pyplot.yscale("log")

# make scatter plot for data (records lost against year)
axes.scatter(years, ssh["compromises"], color="crimson", alpha=1)
axes.scatter(years, mfa["compromises"], color="green", alpha=1)
axes.scatter(years, sms["compromises"], color="darkviolet", alpha=1)
axes.scatter(years, passwords["compromises"], color="darkblue", alpha=1)
axes.set_xticks(years, years)
axes.set_xlabel("Year")
axes.set_ylabel("Number of compromised keys", color="black")
axes.legend(["SSH", "Authenticators", "Calls/SMS", "Passwords"])
axes.tick_params("y", colors="black")

# set plot layout and title
pyplot.title("Number of compromised user secret keys per year, by authentication method")
fig.tight_layout()

# trendline for ssh (quadratic fit)
x1 = [int(k) for k in ssh.year.unique() if [n for n in ssh[ssh.year == k].compromises][0] != 0]
y1 = [k for k in ssh["compromises"] if k != 0]
z1 = numpy.polyfit(x1, y1, 2)
p1 = numpy.poly1d(z1)
pyplot.plot(x1, p1(x1), color="crimson", linestyle="dotted")

# trendline for authenticators (quadratic fit)
x2 = [int(k) for k in mfa.year.unique() if [n for n in mfa[mfa.year == k].compromises][0] != 0]
y2 = [k for k in mfa["compromises"] if k != 0]
z2 = numpy.polyfit(x2, y2, 2)
p2 = numpy.poly1d(z2)
pyplot.plot(x2, p2(x2), color="green", linestyle="dotted")

# first piece of trendline for sms (piecewise linear fit)
sms1 = sms[sms.year < 2022][sms.year != 2017]
x3 = [int(k) for k in sms1.year.unique() if [n for n in sms1[sms1.year == k].compromises][0] != 0]
y3 = [k for k in sms1["compromises"] if k != 0]
j3 = numpy.sin(x3)
z3 = numpy.polyfit(j3, y3, 1)
p3 = numpy.poly1d(z3)
def l3(x):
    # pyplot refused to draw default polyfit, so this
    # numpy equation is manually calculated from polyfit
    return [numpy.exp(3.5*k-7064.5) for k in x]
pyplot.plot(x3, l3(x3), color="darkviolet", linestyle="dotted")

# second piece of trendline for sms (piecewise linear fit)
sms2 = sms[sms.year > 2020]
x4 = [int(k) for k in sms2.year.unique() if [n for n in sms2[sms2.year == k].compromises][0] != 0]
y4 = [k for k in sms2["compromises"] if k != 0]
j4 = numpy.sin(x4)
z4 = numpy.polyfit(j4, y4, 1)
p4 = numpy.poly1d(z4)
def l4(x):
    # pyplot refused to draw default polyfit, so this
    # numpy equation is manually calculated from polyfit
    return [numpy.exp(-1*k+2030) for k in x]
pyplot.plot(x4, l4(x4), color="darkviolet", linestyle="dotted")

# trendline for passwords (linear fit)
x5 = [int(k) for k in passwords.year.unique() if [n for n in passwords[passwords.year == k].compromises][0] != 0]
y5 = [k for k in passwords["compromises"] if k != 0]
z5 = numpy.polyfit(x5, y5, 1)
p5 = numpy.poly1d(z5)
def l5(x):
    # pyplot refused to draw default polyfit, so this
    # numpy equation is manually calculated from polyfit
    return [numpy.exp(0.2*k-382) for k in x]
pyplot.plot(x5, l5(x5), color="darkblue", linestyle="dotted")

# render the figure, display in new window, adjust dimensions and save!
pyplot.show()
