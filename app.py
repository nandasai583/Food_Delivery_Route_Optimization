import streamlit as st
import random
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 🚀 PAGE SETUP
# ----------------------------------------------------
st.set_page_config(page_title="Food Delivery Route Optimizer", layout="wide")
st.title("🍔 Food Delivery Route Optimization Prototype")
st.markdown("""
This prototype demonstrates two routing strategies for food delivery:

- **Greedy Algorithm** – each driver takes the nearest order next.  
- **Advanced Heuristic** – uses clustering and local search (swap-based optimization).

The goal is to minimize total travel distance and improve delivery efficiency.
""")

# ----------------------------------------------------
# 🧭 SIDEBAR SETTINGS
# ----------------------------------------------------
st.sidebar.header("Simulation Settings")
num_restaurants = st.sidebar.slider("Number of Restaurants", 1, 10, 5)
num_orders = st.sidebar.slider("Number of Orders", 5, 50, 20)
num_drivers = st.sidebar.slider("Number of Drivers", 1, 10, 5)
random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

random.seed(random_seed)
np.random.seed(random_seed)

# ----------------------------------------------------
# 🗺️ DATA GENERATION
# ----------------------------------------------------
restaurants = np.random.rand(num_restaurants, 2) * 10
orders = np.random.rand(num_orders, 2) * 10
drivers = np.random.rand(num_drivers, 2) * 10

# ----------------------------------------------------
# 📦 HELPER FUNCTIONS
# ----------------------------------------------------
def euclidean(a, b):
    return np.linalg.norm(a - b)

def greedy_algorithm():
    unassigned = list(range(num_orders))
    driver_routes = [[] for _ in range(num_drivers)]
    driver_distances = np.zeros(num_drivers)
    driver_positions = drivers.copy()

    while unassigned:
        for d in range(num_drivers):
            if not unassigned:
                break
            current = driver_positions[d]
            nearest_idx = min(unassigned, key=lambda i: euclidean(current, orders[i]))
            nearest_order = orders[nearest_idx]
            driver_routes[d].append(nearest_idx)
            driver_distances[d] += euclidean(current, nearest_order)
            driver_positions[d] = nearest_order
            unassigned.remove(nearest_idx)
    return driver_routes, driver_distances.sum()

def advanced_heuristic():
    cluster_labels = [i % num_drivers for i in range(num_orders)]
    random.shuffle(cluster_labels)
    driver_routes = [[] for _ in range(num_drivers)]
    for i, label in enumerate(cluster_labels):
        driver_routes[label].append(i)

    for d in range(num_drivers):
        improved = True
        while improved:
            improved = False
            route = driver_routes[d]
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:]
                    new_route[i], new_route[j] = new_route[j], new_route[i]
                    old_dist = sum(euclidean(orders[route[k]], orders[route[k + 1]]) for k in range(len(route) - 1))
                    new_dist = sum(euclidean(orders[new_route[k]], orders[new_route[k + 1]]) for k in range(len(new_route) - 1))
                    if new_dist < old_dist:
                        driver_routes[d] = new_route
                        improved = True
    total_distance = 0
    driver_positions = drivers.copy()
    for d in range(num_drivers):
        loc = driver_positions[d]
        for o in driver_routes[d]:
            total_distance += euclidean(loc, orders[o])
            loc = orders[o]
    return driver_routes, total_distance

# ----------------------------------------------------
# 🚀 RUN SIMULATION (with session state)
# ----------------------------------------------------
if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

if st.sidebar.button("Generate & Run Simulation"):
    st.session_state.run_clicked = True
    greedy_routes, greedy_total = greedy_algorithm()
    adv_routes, adv_total = advanced_heuristic()
    st.session_state.greedy_routes = greedy_routes
    st.session_state.greedy_total = greedy_total
    st.session_state.adv_routes = adv_routes
    st.session_state.adv_total = adv_total

# ----------------------------------------------------
# ✅ DISPLAY RESULTS
# ----------------------------------------------------
if st.session_state.run_clicked:
    st.subheader("Results Summary")

    results = pd.DataFrame({
        "Algorithm": ["Greedy", "Advanced Heuristic"],
        "Total Distance": [
            st.session_state.greedy_total,
            st.session_state.adv_total
        ]
    })
    st.dataframe(results)

    # 📊 Chart
    fig, ax = plt.subplots()
    ax.bar(results["Algorithm"], results["Total Distance"], color=["skyblue", "lightgreen"])
    ax.set_ylabel("Total Distance (arbitrary units)")
    st.pyplot(fig)

    # 🗺️ Map
    st.subheader("Route Visualization")
    map_center = [5, 5]
    fmap = folium.Map(location=map_center, zoom_start=12)
    colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen", "pink", "gray"]

    for d, route in enumerate(st.session_state.adv_routes):
        color = colors[d % len(colors)]
        points = [list(drivers[d])] + [list(orders[i]) for i in route]
        folium.PolyLine(points, color=color, weight=3, opacity=0.7).add_to(fmap)
        for p in points:
            folium.CircleMarker(location=p, radius=3, color=color, fill=True).add_to(fmap)

    st_folium(fmap, width=800, height=500)

else:
    st.info("Set your parameters and click **Generate & Run Simulation**.")
