

import streamlit as st
import logging
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
import pandas as pd

from langchain.globals import set_verbose
set_verbose(True)

# Azure OpenAI Configuration
# Azure OpenAI Configuration    
api_key = "#######"
api_version = "2023-05-15"
model_name = "gpt-4o"
azure_endpoint = "#####"


llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version,
    model=model_name,
)

# --- Utility Function ---
def calculate_recommended_order_quantity(parameters, sum_mpi_quantity):
    average_daily_use = max(parameters["dailyusage"], 0)
    target_inventory = min(
        max(average_daily_use * parameters["targetinventoryparam"], parameters["targetinventorythreshold"]),
        parameters["maximumquantity"]
    )
    adjusted_quantity = max(0, target_inventory - max(0, sum_mpi_quantity))
    recommended_quantity = max(1, round(adjusted_quantity / parameters["eachesquantity"]))
    logging.info(f"Calculated recommended order quantity: {recommended_quantity}")
    return recommended_quantity

# --- LangChain Tools ---
@tool(description="Simulates inventory for next 30 days")
def simulate_inventory(currentinventory: int, dailyusage: float, reorderpoint: int, orderquantity: int, leadtime: int, numdays: int,
                       picked: int, restocked: int, targetinventorythreshold: float, targetinventoryparam: float,
                       maximumquantity: int, eachesquantity: int) -> str:
    
    new_inventory = currentinventory + restocked - picked
    parameters = {
        "dailyusage": dailyusage,
        "targetinventoryparam": targetinventoryparam,
        "targetinventorythreshold": targetinventorythreshold,
        "maximumquantity": maximumquantity,
        "eachesquantity": eachesquantity
    }
    sum_mpi_quantity = 3  # Placeholder
    recommended_order = calculate_recommended_order_quantity(parameters, sum_mpi_quantity)

    prompt = f"""
    You are an AI assistant helping with inventory management. Simulate inventory data for the next {numdays} days using the following parameters:
    - Starting inventory: {new_inventory}
    - Daily usage: {dailyusage}
    - Reorder point: {reorderpoint}
    - Order quantity: {orderquantity}
    - Lead time: {leadtime}
    - Recommended order quantity: {recommended_order}

    For each day, provide:
    1. Day number
    2. Remaining inventory after daily usage
    3. Indicate if an order is placed (yes/no)
    4. The inventory level after any incoming orders are delivered
    5. New inventory (calculated as starting_inventory + restocked - picked)
    6. Recommended Order Quantity

    Return a markdown table with columns: "Day", "Inventory Level", "Order Placed", "Post-Order Inventory", "New Inventory", "Recommended Order Quantity".
    """
    return llm.predict(prompt)

@tool(description="Analyze Simulation output ")
def analyze_simulation(simulation_data: str) -> str:
    prompt = f"""
    Based on the following inventory simulation data:
    {simulation_data}

    Analyze and explain:
    - Stockouts and when they occurred
    - Most optimal reorder day
    - Inventory fluctuation trends
    - Recommended inventory management adjustments
    - If recommended order quantity aligns with demand
    - Suggest top 5 combinations of picked and restocked that improve inventory control
    - Justify picked/restocked logic
    - How the top combinations will help over the next 30 days
    """
    return llm.predict(prompt)

# --- Agent Initialization ---
agent = initialize_agent(
    tools=[simulate_inventory, analyze_simulation],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# --- Streamlit UI ---
st.set_page_config(page_title="Inventory Agent", layout="wide")
st.title("Inventory Simulation and Analysis Agent")

#Streamlit Form with Submit Button
with st.form("input_form"):
    st.subheader("Enter Inventory Parameters")

    currentinventory = st.number_input("Current Inventory", value=79)
    dailyusage = st.number_input("Daily Usage", value=3.3182, format="%.4f")
    reorderpoint = st.number_input("Reorder Point", value=22)
    orderquantity = st.number_input("Order Quantity", value=1)
    leadtime = st.number_input("Lead Time (days)", value=5)
    numdays = st.number_input("Simulation Days", value=30)
    picked = st.number_input("Picked Quantity", value=3)
    restocked = st.number_input("Restocked Quantity", value=3)
    targetinventorythreshold = st.number_input("Target Inventory Threshold", value=7.0, format="%.2f")
    targetinventoryparam = st.number_input("Target Inventory Multiplier", value=1.2, format="%.2f")
    maximumquantity = st.number_input("Maximum Inventory Quantity", value=108)
    eachesquantity = st.number_input("Eaches per Order Unit", value=36)

    submitted = st.form_submit_button("Run Simulation")

#Handle Submission
if submitted:
    parameters = {
        "currentinventory": currentinventory,
        "dailyusage": dailyusage,
        "reorderpoint": reorderpoint,
        "orderquantity": orderquantity,
        "leadtime": leadtime,
        "numdays": numdays,
        "picked": picked,
        "restocked": restocked,
        "targetinventorythreshold": targetinventorythreshold,
        "targetinventoryparam": targetinventoryparam,
        "maximumquantity": maximumquantity,
        "eachesquantity": eachesquantity,
    }

    prompt = f"Simulate inventory using these parameters:\n{parameters}"
    with st.spinner("Running Inventory Simulation..."):
        simulation_result = agent.invoke(prompt)
        st.markdown("Simulation Output")
        st.markdown(simulation_result)

        with st.spinner("Analyzing Simulation..."):
            insight_prompt = f"Analyze simulation data:\n{simulation_result}"
            insight_result = agent.invoke(insight_prompt)
            st.markdown("### Analysis & Recommendations")
            st.markdown(insight_result)
