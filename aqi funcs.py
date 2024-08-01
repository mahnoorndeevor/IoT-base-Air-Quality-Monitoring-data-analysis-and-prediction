def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0

df["CO_SubIndex"] = df["CO"].apply(lambda x: get_CO_subindex(x))

def get_CH4_subindex(x):
    if x <= 2:
        return x * 50 / 2
    elif x <= 4:
        return 50 + (x - 2) * 50 / 2
    elif x <= 10:
        return 100 + (x - 4) * 100 / 6
    elif x <= 20:
        return 200 + (x - 10) * 100 / 10
    elif x <= 40:
        return 300 + (x - 20) * 100 / 20
    elif x > 40:
        return 400 + (x - 40) * 100 / 40
    else:
        return 0
df["Methane_SubIndex"] = df["Methane"].apply(lambda x: get_CH4_subindex(x))


def get_CO2_subindex(x):
    if x <= 400:
        return x * 50 / 400
    elif x <= 800:
        return 50 + (x - 400) * 50 / 400
    elif x <= 1200:
        return 100 + (x - 800) * 100 / 400
    elif x <= 1800:
        return 200 + (x - 1200) * 100 / 600
    elif x <= 2400:
        return 300 + (x - 1800) * 100 / 600
    elif x > 2400:
        return 400 + (x - 2400) * 100 / 1200
    else:
        return 0
df["CO2_SubIndex"] = df["CO2"].apply(lambda x: get_CO2_subindex(x))



print(df.head())

#print(df["CO_SubIndex"].min())
