import streamlit as st
import pandas as pd
import numpy as np
import googlemaps
from ortools.linear_solver import pywraplp

st.title("🛠️ Service Scheduler with OR-Tools and Google Maps (Normalized Costs)")

# Sidebar config
# api_key = st.sidebar.text_input("Google Maps API Key", type="password")

api_key = 'AIzaSyC74qA0gzDYaM76ESu7uUX7sPT979nyNGw'
skill_weight = st.sidebar.slider("Skill Mismatch Weight", 1, 10, 3)
time_weight = st.sidebar.slider("Travel Time Weight", 1, 10, 1)
max_customers_per_rep = st.sidebar.number_input("Max Customers Per Rep", min_value=1, max_value=10, value=3)

uploaded_file = pd.read_excel("service.xlsx")

def compute_skill_mismatch(cust_row, rep_row, skills):
    mismatch = 0
    for skill in skills:
        mismatch += abs(cust_row[skill] - rep_row[skill])
    return mismatch

def get_travel_times(gmaps, reps, customers):
    origins = list(zip(reps['Latitude'], reps['Longitude']))
    destinations = list(zip(customers['Latitude'], customers['Longitude']))
    origins_str = [f"{lat},{lng}" for lat, lng in origins]
    destinations_str = [f"{lat},{lng}" for lat, lng in destinations]

    result = gmaps.distance_matrix(origins_str, destinations_str, mode="driving")
    travel_times = []
    for row in result['rows']:
        times = []
        for el in row['elements']:
            if el['status'] == 'OK':
                times.append(el['duration']['value'] / 60)  # convert seconds to minutes
            else:
                times.append(9999)  # unreachable penalty
        travel_times.append(times)
    return np.array(travel_times)

if True :
    gmaps = googlemaps.Client(key=api_key)

    # Load sheets
    try:
       
        customers = pd.read_excel("service.xlsx", sheet_name="Customers")
        reps = pd.read_excel("service.xlsx", sheet_name="Reps")
    except Exception as e:
        st.error(f"Error reading sheets: {e}")
        st.stop()

    st.subheader("Customers")
    st.dataframe(customers)

    st.subheader("Reps")
    st.dataframe(reps)

    skill_cols = ['Electrical', 'HVAC', 'Plumbing', 'Software']

    # Check required columns exist
    for df, name in [(customers, "Customers"), (reps, "Reps")]:
        for col in ['Latitude', 'Longitude'] + skill_cols:
            if col not in df.columns:
                st.error(f"Missing '{col}' in {name} sheet.")
                st.stop()

    # Calculate travel times
    st.info("Calculating travel times (Google Maps)...")
    travel_times = get_travel_times(gmaps, reps, customers)
    st.write("Travel time matrix (minutes):")
    st.dataframe(pd.DataFrame(travel_times, index=reps['RepID'], columns=customers['CustomerID']))

    # Precompute skill mismatches matrix
    skill_mismatches = np.zeros((len(reps), len(customers)))
    for i in range(len(reps)):
        for j in range(len(customers)):
            skill_mismatches[i, j] = compute_skill_mismatch(customers.iloc[j], reps.iloc[i], skill_cols)

    max_skill_mismatch = skill_mismatches.max()
    max_travel_time = travel_times[travel_times < 9999].max()  # ignore unreachable penalty

    # Normalize to 0-1
    norm_skill_mismatches = skill_mismatches / max_skill_mismatch if max_skill_mismatch > 0 else skill_mismatches
    norm_travel_times = travel_times / max_travel_time if max_travel_time > 0 else travel_times

    st.write("Normalized Skill Mismatch matrix:")
    st.dataframe(pd.DataFrame(norm_skill_mismatches, index=reps['RepID'], columns=customers['CustomerID']))

    st.write("Normalized Travel Time matrix:")
    st.dataframe(pd.DataFrame(norm_travel_times, index=reps['RepID'], columns=customers['CustomerID']))

    # Setup OR-Tools solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        st.error("Could not create OR-Tools solver instance.")
        st.stop()

    x = {}
    for i in range(len(reps)):
        for j in range(len(customers)):
            x[i, j] = solver.BoolVar(f'x[{i},{j}]')

    # Each customer assigned exactly to one rep
    for j in range(len(customers)):
        solver.Add(solver.Sum(x[i, j] for i in range(len(reps))) == 1)

    # Each rep can handle up to max_customers_per_rep customers
    for i in range(len(reps)):
        solver.Add(solver.Sum(x[i, j] for j in range(len(customers))) <= max_customers_per_rep)

    # Objective: minimize weighted sum of normalized skill mismatch and normalized travel time
    objective = solver.Objective()

    for i in range(len(reps)):
        for j in range(len(customers)):
            cost = skill_weight * norm_skill_mismatches[i][j] + time_weight * norm_travel_times[i][j]
            objective.SetCoefficient(x[i, j], cost)

    objective.SetMinimization()

    st.info("Solving the assignment problem...")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        st.success("Optimal assignment found!")
        assignments = []
        for i in range(len(reps)):
            for j in range(len(customers)):
                if x[i, j].solution_value() > 0.5:
                    assignments.append({
                        "RepID": reps.loc[i, 'RepID'],
                        "CustomerID": customers.loc[j, 'CustomerID'],
                        "Skill Mismatch": round(skill_mismatches[i, j], 2),
                        "Travel Time (min)": round(travel_times[i, j], 2),
                        "Weighted Cost": round(skill_weight * norm_skill_mismatches[i][j] + time_weight * norm_travel_times[i][j], 3)
                    })
        st.subheader("Assignments")
        st.dataframe(pd.DataFrame(assignments))
    else:
        st.error("No optimal assignment found. Try increasing max customers per rep or check data validity.")
else:
    st.info("Upload the Excel file and enter your Google Maps API key to start.")
