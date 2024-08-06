"""
This code is used to illustrate a simple challenge when using a linear cost function to determine
the least cost path trough a graph: that we tend to get corner solutions.

See the README file for further explanation. 
"""

# Import necessary libraries
import numpy as np
import pandas as pd

# Define the function to calculate the objective for a given path and choice of alphas
def get_objective(df, links, constraint, alpha1, alpha2, objective_type = "nonlinear"):
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
        if objective_type == "nonlinear":
            objective = (alpha1 * total_cost) - (alpha2 * np.sqrt(remaining_constraint))
        else:
            objective = (alpha1 * total_cost) - (alpha2 * remaining_constraint)

        # Print
        # print(f"Link: {link}, Total cost: {total_cost}, Boost: {boost}, Remaining Constraint: {remaining_constraint}, Objective: {objective}")

    return objective

# Define the graph
links = [1, 2, 3, 4, 5]
cost = [1, 1, 1.5, 1, 1.5]
boost = [0, 0, 1, 0, 1]
type = ["Road", "Road", "Charge", "Road", "Charge"]

# Define the graph and show it
graph = pd.DataFrame({"links": links, "cost": cost, "boost": boost, "type": type})
print(graph)

# Set the constraint
constraint = 3

# Define the choice the user must make. In this case, they must take Link 1.
# They then choose between Links 2 and 3, and then between Links 4 and 5.
choices = [[1], [2, 3], [4, 5]]

# Define the function to get the user's selections for a given alpha1
def get_selections(alpha1, objective_type):
    selections = []

    alpha2 = 1-alpha1

    for choice in choices:
        obj_values = [get_objective(graph, selections + [link], constraint, alpha1, alpha2, objective_type) for link in choice]
        min = np.argmin(obj_values)
        selections.append(choice[min])
    
    return selections

print("\nAlpha 1: User's distate for taking longer time, Alpha 2: User's distate for running out of energy")
print("Results using a linear Cost Function")
for alpha in np.linspace(0, 1, 11):
    selections = get_selections(alpha, "linear")
    corner_solution = [2 in selections and 4 in selections][0] or [3 in selections and 5 in selections][0]
    type_of_corner = "Double road" if [2 in selections and 4 in selections][0] else "Double charging" if [3 in selections and 5 in selections][0] else "None"
    print(f"alpha1: {alpha:.2f}, alpha2: {1-alpha:.2f}, Selections: {selections}{''if not corner_solution else '<- Corner Solution'}{(f' ({type_of_corner})' if corner_solution else '')}")

print("\nResults using a Nonlinear Cost Function")
for alpha in np.linspace(0, 1, 11):
    selections = get_selections(alpha, "nonlinear")
    corner_solution = [2 in selections and 4 in selections][0] or [3 in selections and 5 in selections][0]
    type_of_corner = "Double road" if [2 in selections and 4 in selections][0] else "Double charging" if [3 in selections and 5 in selections][0] else "None"
    print(f"alpha1: {alpha:.2f}, alpha2: {1-alpha:.2f}, Selections: {selections}{''if not corner_solution else '<- Corner Solution'}{(f' ({type_of_corner})' if corner_solution else '')}")





