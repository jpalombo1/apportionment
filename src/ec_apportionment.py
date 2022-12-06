from pathlib import Path
from typing import List

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore


def proportion_method(
    num_states: int, num_seats: int, populations: NDArray[np.floating]
) -> NDArray[np.integer]:
    """Use proportion method where state seats = rem_seats * state_pop / total_pop + 3 constant (2 senate, 1 gov).

    Args:
        num_seats (int): Number of total EC eats to give out.
        num_states (int): Number of total states to split between.
        populations (NDArray[np.floating]): Array of populations per state.

    Returns:
        NDArray[np.integer]: Array of proportional seats per state based on populations.
    """
    CONSTANT_SEATS = 3
    remSeats = num_seats - num_states
    total_population = populations.sum()
    return np.around(populations / total_population * remSeats + CONSTANT_SEATS).astype(
        int
    )


def predict_error(
    pred_seats: NDArray[np.integer],
    states: List[str],
    ec_votes: NDArray[np.integer],
    model_name: str,
) -> None:
    """Calculate metrics by prediction error of predicted portion vs actual EC votes. Log missed states.

    Args:
        pred_seats (NDArray[np.integer]): Predicted allocated seats per state by given model.
        states (List[str]): String names of states.
        ec_votes (NDArray[np.integer]): Actual EC votes designated per state.
        model_name (str): Name of model used.
    """
    errors = 0
    for state, ec_pred, ec_actual in zip(states, pred_seats, ec_votes):
        if ec_pred != ec_actual:
            print(
                f"Mismatch : State {state} Actual {ec_actual} Guess {ec_pred} Diff: {np.abs(ec_actual - ec_pred)}"
            )
            errors += 1
        else:
            print(f"Correct!, State {state}")

    print(
        f"{model_name} model is {(1 - errors / len(states)) * 100:.2f}% percent correct"
    )


def state_most_wins(
    num_states: int, populations: NDArray[np.floating], state_seats: NDArray[np.integer]
) -> int:
    """Get most wins in main bulk of apportionment alg. Explain more...

    Args:
        num_states (int): Number of total states to split between.
        populations (NDArray[np.floating]): Array of populations per state.
        state_seats (NDArray[np.integer]): Array of currently designated state seats

    Returns:
        int: State index that had the most wins.
    """
    wins = np.zeros(num_states)
    for sa in range(num_states):
        A_pop_per_rep = populations[sa] / state_seats[sa]
        A_pop_per_rep_add = populations[sa] / (state_seats[sa] + 1)
        for sb in range(num_states):
            if sb != sa:
                B_pop_per_rep = populations[sb] / state_seats[sb]
                B_pop_per_rep_add = populations[sb] / (state_seats[sb] + 1)

                A_vs = np.abs(A_pop_per_rep_add - B_pop_per_rep) / (A_pop_per_rep_add)
                vs_A = np.abs(A_pop_per_rep - B_pop_per_rep_add) / (B_pop_per_rep_add)
                if A_vs < vs_A:
                    wins[sa] += 1
    return np.argmax(wins).astype(int)


def apportionment_method(
    num_seats: int,
    num_states: int,
    populations: NDArray[np.floating],
) -> NDArray[np.integer]:
    """Apportionment algorithm.
    For first seats, automatically apportion 1 state seat and 3 EC seats per state.
    Then designate each next seat to state that has most wins for all the rest of seats.

    Args:
        num_seats (int): Number of total EC eats to give out.
        num_states (int): Number of total states to split between.
        populations (NDArray[np.floating]): Array of populations per state.

    Returns:
        NDArray[np.integer]: Array of apportioned seats per state.
    """
    CONSTANT_STATE_SEATS = 1
    CONSTANT_EC_SEATS = 3
    state_seats = np.zeros(num_states, dtype=int)
    ec_seats = np.zeros(num_states, dtype=int)
    state_seats[:num_seats] = CONSTANT_STATE_SEATS
    ec_seats[:num_seats] = CONSTANT_EC_SEATS
    for _ in tqdm(range(num_states, num_seats + 1)):
        most_wins = state_most_wins(
            num_states=num_states, populations=populations, state_seats=state_seats
        )
        state_seats[most_wins] += 1
        ec_seats[most_wins] += 1

    return ec_seats


def main():
    """Main function."""
    NUM_STATES = 51
    NUM_SEATS = 435
    CSV_PATH = Path(__file__).parent.parent / "data" / "census_state_info.csv"

    df = pd.read_csv(CSV_PATH)
    states = df["State"].tolist()
    populations = df["Population"].to_numpy()
    ec_votes = df["EC Votes"].to_numpy()

    prop_seats = proportion_method(
        num_seats=NUM_SEATS, num_states=NUM_STATES, populations=populations
    )
    predict_error(prop_seats, states, ec_votes, "Proportional")

    ec_seats = apportionment_method(
        num_seats=NUM_SEATS,
        num_states=NUM_STATES,
        populations=populations,
    )
    predict_error(ec_seats, states, ec_votes, "Apportionment")


if __name__ == "__main__":
    main()
