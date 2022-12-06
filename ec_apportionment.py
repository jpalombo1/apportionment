import numpy as np
import pandas as pd
from tqdm import tqdm  # type: ignore

numStates = 51
numSeats = 435

stateSeats = {}
ECSeats = {}

df = pd.read_csv("data/census_state_info.csv")
# print(df)
states = df["State"]
populations = df["Population"]

remSeats = numSeats - numStates
total_population = populations.sum()
propSeats = round(populations / total_population * remSeats + 3)  # type: ignore


errors = 0
for e, ecseat in enumerate(propSeats):
    ec_actual = df["EC Votes"][e]
    if ecseat != ec_actual:
        print(
            "Mismatch : State ",
            states[e],
            ", actual",
            ec_actual,
            ", guess ",
            int(ecseat),
            ", diff ",
            np.abs(ec_actual - int(ecseat)),
        )
        errors += 1
    else:
        print("Correct!, State", states[e])

if errors > 0:
    print(
        "Proportional model is ", (1 - errors / len(states)) * 100, " percent correct"
    )
else:
    print("Proportional model is perfect!")


for seat in tqdm(range(numSeats + 1)):
    if seat < numStates:
        stateSeats[states[seat]] = 1
        ECSeats[states[seat]] = 3
    else:
        wins = np.zeros(len(states))
        for sa, stateA in enumerate(states):
            A_pop_per_rep = populations[sa] / stateSeats[stateA]
            A_pop_per_rep_add = populations[sa] / (stateSeats[stateA] + 1)
            for sb, stateB in enumerate(states):
                if sb != sa:
                    B_pop_per_rep = populations[sb] / stateSeats[stateB]
                    B_pop_per_rep_add = populations[sb] / (stateSeats[stateB] + 1)

                    A_vs = np.abs(A_pop_per_rep_add - B_pop_per_rep) / (
                        A_pop_per_rep_add
                    )
                    vs_A = np.abs(A_pop_per_rep - B_pop_per_rep_add) / (
                        B_pop_per_rep_add
                    )
                    if A_vs < vs_A:
                        wins[sa] += 1

        most_wins = np.argmax(wins)
        stateSeats[states[most_wins]] += 1  # type: ignore
        ECSeats[states[most_wins]] += 1  # type: ignore

print(ECSeats)

errors = 0
for state in ECSeats:
    ec_actual = int(df[df["State"] == state]["EC Votes"])
    if ECSeats[state] != ec_actual:
        print(
            "Mismatch : State ",
            state,
            ", actual",
            ec_actual,
            ", guess ",
            ECSeats[state],
            ", diff",
            np.abs(ec_actual - ECSeats[state]),
        )
        errors += 1
    else:
        print("Correct!, State", state)

if errors > 0:
    print(
        "Approtionment model is ", (1 - errors / len(states)) * 100, " percent correct"
    )
else:
    print("Apportionment model is perfect!")
