import numpy as np
import pandas as pd


def get_objective(df, links, constraint, alpha1, alpha2):
    objective = 0
    total_cost = 0
    remaining_constraint = constraint

    for link in links:
        # Get costs and boost
        cost  = df.loc[df["links"] == link, "cost"].values[0]
        total_cost += cost
        boost = df.loc[df["links"] == link, "boost"].values[0]

        # Calculate the remaining constraint
        remaining_constraint = remaining_constraint - cost + boost

        # Calculate the objective
        objective = (alpha1 * total_cost) - (alpha2 * np.sqrt(remaining_constraint))

        # Print
        # print(f"Link: {link}, Total cost: {total_cost}, Boost: {boost}, Remaining Constraint: {remaining_constraint}, Objective: {objective}")

    return objective


links = [1, 2, 3, 4, 5]
cost = [1, 1, 1.5, 1, 1.5]
boost = [0, 0, 1, 0, 1]

graph = pd.DataFrame({"links": links, "cost": cost, "boost": boost})

constraint = 3

choices = [[1], [2, 3], [4, 5]]


def get_selections(alpha1):
    selections = []

    alpha2 = 1-alpha1

    for choice in choices:
        obj_values = [get_objective(graph, selections + [link], constraint, alpha1, alpha2) for link in choice]
        min = np.argmin(obj_values)
        selections.append(choice[min])
    
    return selections

for alpha in np.linspace(0, 1, 11):
    print(f"alpha1: {alpha:.2f}, alpha2: {1-alpha:.2f}, Selections: {get_selections(alpha)}")






