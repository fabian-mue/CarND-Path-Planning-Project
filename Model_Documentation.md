# CarND-Path-Planning-Project
## Goals
In this project the goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. For this the e car's localization and sensor fusion data are provided. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another and at this the car should be able to make one complete loop around the 6946m highway. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

Finally, it was my task to add the code model which should use the provided information to control the ego vehicle properly to navigate it around the highway meeting all stated goals and requirements.

## Code Model
In order to navigate the car save around the virtual highway this basic model of actions is applied: 
Prediction  -> Behavioral Planning -> Trajectory Generation (see image below).
<img src="/home/fabian/CarND-Path-Planning-Project/model_overview.png" style="zoom:25%;" />

In the following, these steps and their implementation in the code are described more in detail.
Due to all the implementation has been done in src/main.cpp, all the upcoming comments regarding to the code refer to this file.

### Prediction 

The main functionality for prediction is implemented in the function "Predict_TrafficInformation" (line 137 -181). It uses the information about the other cars from the fusion sensor to predict if there will be another car on the right/left side or ahead within a critical distance regarding to the s-coordinates. This information should keep the ego vehicle away from risky driving maneuvers or enable it to accelerate as well as to change the lane. At this it evaluates the next position of the other vehicle by using its speed information for x and y, checks the current lane of the other vehicle by evaluating its d-coordinate and makes sure that the distance between the s-position of the other vehicle and the ego vehicle is not lower than the critical distance of 30.
The function "Predict_TrafficInformation" is called in line 459 for each other car whose information is stored in the sensor_fusion array.

### Behavioral Planning

The main functionality for behavioral planning is implemented in the function "UpdateVehicleBehavior" (line 184 - 224). It uses the gathered predictions about the behavior of the other cars and evaluates them. If there is an other car ahead, the ego vehicle will try to switch the lane depending on if a lane change is possible in general and if the lane to the right/left side is free so that no collision will be caused. In the case no lane switch is possible, the ego vehicle needs to break to avoid a collision.  The break is implemented by subtracting 0.224 mph from the reference speed. 
If there is no vehicle ahead, the ego does not drive faster than 49.276 mph it accelerates by adding 0.224 mph to the reference speed. In general, the ego vehicle shall also always try to be on the middle lane which is an efficient approach to drive fast because then the vehicle is able to make right and left overtake maneuvers.
Moreover, the reference speed is used to specify how fast the ego vehicle has to drive so if possible it should be around 49.5 mph.
The function "UpdateVehicleBehavior" is called in line 469 using the localization data of the ego vehicle and the previously predicted information.

### Trajectory Generation

The main functionality for behavioral planning is implemented in the function "GenerateTrajectory" (line 227 - 323). 

First some references values are retrieved depending on the length of the previous path (line 240). For this the function "GetReferenceValues" is used (line 326-354).

The trajectories are generated by using the points from the previous path which the car has not traveled yet and inserting new points. The x values of this points are generated by defining a distance which is sampled so that the car drives the required reference speed. Additionally, a spline is generated to connect this points and provide the corresponding y values which enables a smooth driving path for the ego vehicle. The spline generation is implemented in the provided "spline.h" library. Previously, the spline has been set by defining several equally 30m spaced points.

Additionally, during these calculations a coordinate transformation to car coordinates takes place to simplify the applied math (line 263).  In the end a rotation back to the global coordinate system takes is done (line 310).

The function "GenerateTrajectory" is called in line 474 using the localization data of the ego vehicle, the previous path and the way points information.



###  Conclusion 

In total this was a very interesting project. 
It clearly shows how important a reliable interaction between prediction, behavior planning and trajectory generation is to allow a car to drive safely and autonomously on a highway. 
In a city with additional degrees of freedom for driving maneuvers, this would certainly be even more challenging and more complex approaches such as machine learning or more extensive cost functions would be required. 
