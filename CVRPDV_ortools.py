"""Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).
   ortools==6.10.6025
""" 
from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import pandas as pd
import time
import numpy as np
from Parameter import num,url,v_trans,Q,c_d,f,h1,h2,rawdata,D,T,time_windows,location,demands

###########################
# Problem Data Definition #
###########################
def create_data_model(num=num,url=url):
  """Stores the data for the problem"""
  data = {}
  
  _locations = location
  
  # Multiply coordinates in block units by the dimensions of an average city block, 114m x 80m,
  # to get location coordinates.
  data["locations"] = [(l[0], l[1]) for l in _locations]
  data["num_locations"] = len(data["locations"])
  num_vehicles=50
  data["num_vehicles"] = num_vehicles
  data["depot"] = 0
  data["demands"] = demands
  capacities = []
  for i in range(num_vehicles):
      capacities.append(12)
  data["vehicle_capacities"] = capacities
  data["time_windows"] = time_windows #min
  data["time_per_demand_unit"] = 0.5
  data["vehicle_speed"] = 500 #m/min 即30km/h
  return data
#######################
# Problem Constraints #
#######################
def manhattan_distance(position_1, position_2):
  """Computes the Manhattan distance between two points"""
  return (abs(position_1[0] - position_2[0]) + abs(position_1[1] - position_2[1]))*300

def create_distance_callback(data):
  """Creates callback to return distance between points."""
  _distances = {}

  for from_node in range(data["num_locations"]):
    _distances[from_node] = {}
    for to_node in range(data["num_locations"]):
      if from_node == to_node:
        _distances[from_node][to_node] = 0
      else:
        _distances[from_node][to_node] = (
            manhattan_distance(data["locations"][from_node],
                               data["locations"][to_node]))

  def distance_callback(from_node, to_node):
    """Returns the manhattan distance between the two nodes"""
    return _distances[from_node][to_node]

  return distance_callback

def create_demand_callback(data):
  """Creates callback to get demands at each location."""
  def demand_callback(from_node, to_node):
    return data["demands"][from_node]
  return demand_callback


def add_capacity_constraints(routing, data, demand_evaluator):
  """Adds capacity constraint"""
  capacity = "Capacity"
  routing.AddDimensionWithVehicleCapacity(
      demand_evaluator,
      0, # null capacity slack
      data["vehicle_capacities"], # vehicle maximum capacities
      True, # start cumul to zero
      capacity)

def create_time_callback(data):
  """Creates callback to get total times between locations."""
  def service_time(node):
    """Gets the service time for the specified location."""
    return round(data["demands"][node] * data["time_per_demand_unit"],0)

  def travel_time(from_node, to_node,h1=0.3,h2=0.15):
    """Gets the travel times between two locations."""
     #h1=0.3 caution intensity when depart from depot
     #h2=0.15 caution intensity in-transit
    if from_node == to_node:
      travel_time = 0
    elif from_node==(5,5):
        travel_time = manhattan_distance(
                data["locations"][from_node],
                data["locations"][to_node]) / (data["vehicle_speed"]*(1-h1))
    elif to_node==(5,5):
        travel_time = manhattan_distance(
                data["locations"][from_node],
                data["locations"][to_node]) / (data["vehicle_speed"]*(1-0))
    else:
        travel_time = manhattan_distance(
                data["locations"][from_node],
                data["locations"][to_node]) / (data["vehicle_speed"]*(1-h2))
    return travel_time

  def time_callback(from_node, to_node):
    """Returns the total time between the two nodes"""
    serv_time = service_time(from_node)
    trav_time = travel_time(from_node, to_node)
    return serv_time + trav_time

  return time_callback
def add_time_window_constraints(routing, data, time_callback):
  """Add Global Span constraint"""
  time = "Time"
  horizon = 300
  routing.AddDimension(
    time_callback,
    horizon, # allow waiting time
    horizon, # maximum time per vehicle
    False, # Don't force start cumul to zero. This doesn't have any effect in this example,
           # since the depot has a start window of (0, 0).
    time)
  time_dimension = routing.GetDimensionOrDie(time)
  for location_node, location_time_window in enumerate(data["time_windows"]):
        index = routing.NodeToIndex(location_node)
        time_dimension.CumulVar(index).SetRange(location_time_window[0], location_time_window[1])

###########
# Printer #
###########
def print_solution(data, routing, assignment):
  """Prints assignment on console"""
  # Inspect solution.
  capacity_dimension = routing.GetDimensionOrDie('Capacity')
  time_dimension = routing.GetDimensionOrDie('Time')
  total_dist = 0
  time_matrix = 0
  plan_routings=[]
  plan_delivery_time=[]
  route_dis=[]

  for vehicle_id in range(data["num_vehicles"]):
    plan_node=[]
    delivery_time=[]
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
    route_dist = 0
    while not routing.IsEnd(index):
      node_index = routing.IndexToNode(index)
      next_node_index = routing.IndexToNode(
        assignment.Value(routing.NextVar(index)))
      route_dist += manhattan_distance(
        data["locations"][node_index],
        data["locations"][next_node_index])
      load_var = capacity_dimension.CumulVar(index)
      route_load = assignment.Value(load_var)
      time_var = time_dimension.CumulVar(index)
      time_min = assignment.Min(time_var)
      time_max = assignment.Max(time_var)
      plan_node.append(node_index)
      delivery_time.append(time_min)
      plan_output += ' {0} Load({1}) Time({2},{3}) ->'.format(
        node_index,
        route_load,
        time_min, time_max)
      
      index = assignment.Value(routing.NextVar(index))

    node_index = routing.IndexToNode(index)
    load_var = capacity_dimension.CumulVar(index)
    route_load = assignment.Value(load_var)
    time_var = time_dimension.CumulVar(index)
    route_time = assignment.Value(time_var)
    time_min = assignment.Min(time_var)
    time_max = assignment.Max(time_var)
    total_dist += route_dist
    time_matrix += route_time
    delivery_time.append(time_min)
    plan_output += ' {0} Load({1}) Time({2},{3})\n'.format(node_index, route_load,
                                                           time_min, time_max)
    plan_output += 'Distance of the route: {0} m\n'.format(route_dist)
    plan_output += 'Load of the route: {0}\n'.format(route_load)
    plan_output += 'Time of the route: {0} min\n'.format(route_time)
    route_dis.append(route_dist)
    plan_routings.append(plan_node)
    plan_delivery_time.append(delivery_time)
    # print("The nodes in the vehicle route:",plan_node)
    # print("The delivery_time at each node in the vehicle route:",delivery_time)
    
    # print(plan_output)
    # print(plan_delivery_time)
 
  vehicle_routing=[]
  for i in plan_routings:
      if len(i)==1:
          continue
      else:
          for j in i:
              vehicle_routing.append(j)
  vehicle_routing.append(0)
  batch_delivery_time=[]
  for i in plan_delivery_time:
      if len(i)==2:
          continue
      else:
          batch_delivery_time.append(i[-2])
  batch_route_delivery_time=[]
  for i in plan_delivery_time:
      if len(i)==2:
          continue
      else:
          batch_route_delivery_time.append(i[-1])
  # print("vehicle_routing:",vehicle_routing)
  batch_route_dis=[]
  for i in route_dis:
      if i==0:
          continue
      else:
          batch_route_dis.append(i)
  # print("batch_route_dis",batch_route_dis)
  # print('Total Distance of all routes: {0} m'.format(total_dist))
  # print('Total Time of all routes: {0} min'.format(time_matrix))
  # print('Total cost of all routes: {0} RMB'.format(total_dist)
  return vehicle_routing,plan_delivery_time,batch_delivery_time,batch_route_delivery_time,batch_route_dis

########
# Main #
########
def main(num,url):
  """Entry point of the program"""
  # Instantiate the data problem.
  data = create_data_model(num,url)
  # Create Routing Model
  routing = pywrapcp.RoutingModel(data["num_locations"], data["num_vehicles"], data["depot"])
  # Define weight of each edge
  distance_callback = create_distance_callback(data)
  routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
  # Add Capacity constraint
  demand_callback = create_demand_callback(data)
  add_capacity_constraints(routing, data, demand_callback)
  # Add Time Window constraint
  time_callback = create_time_callback(data)
  add_time_window_constraints(routing, data, time_callback)

  # Setting first solution heuristic (cheapest addition).
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  # Solve the problem.
  assignment = routing.SolveWithParameters(search_parameters)
  if assignment:
    vehicle_routing,plan_delivery_time,batch_delivery_time,batch_route_delivery_time,batch_route_dis= print_solution(data, routing, assignment)
  return vehicle_routing

# if __name__ == '__main__':
#   start_time=time.time()
#   vehicle_routing=main()
#   print("vehicle_routing:",vehicle_routing)
#   end_time=time.time()
#   used_time = end_time - start_time
#   print("CPU Time used:", used_time)
 
vehicle_routing_OR=main(num,url)
