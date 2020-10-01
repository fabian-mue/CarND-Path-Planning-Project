#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "spline.h"
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"

#define NUMBER_OF_CYCLES_PER_SEC (50U)

#define CYCLE_TIME (0.02) // in milliseconds
#define POINTS_PER_PATH (50U)
#define LEFT_LANE_BORDER_LEFT (0U)
#define LEFT_LANE_BORDER_RIGHT (4U)
#define RIGHT_LANE_BORDER_LEFT (8U)
#define RIGHT_LANE_BORDER_RIGHT (12U)
#define CRITICAL_DIST_TO_EGO_VEHICLE (30.0) // in meters
#define NEXT_POINTS_SPACE (30U) // in meters
#define MAX_SPEED (49.5) // in mph
#define SPEED_DELTA (0.224) // used for breaking/acceleration, ends up to be under 10 m/s^2 requirement


// for convenience
using nlohmann::json;
using std::string;
using std::vector;
using std::cout;
using std::endl;

enum Lane_e
{
    LEFT_LANE = 0U,
    MIDDLE_LANE = 1U,
    RIGHT_LANE = 2U,
    NO_LANE = 255U,
};

struct Traffic_Information_t
{
  bool CarRight_b;
  bool CarLeft_b;
  bool CarAhead_b;
};

struct Sensor_Fusion_Data_t
{
  uint32_t ID_u32;
  double x;
  double y;
  double vx;
  double vy;
  double s;
  double d;
};

struct Localization_Data_t
{
  double x;
  double y;
  double s;
  double d;
  double yaw;
  double v;
  double v_ref;
  Lane_e lane;
  uint16_t previousPathSize;
};

struct Reference_Information_t
{
  double x_prev;
  double y_prev;
  double x;
  double y;
  double yaw;
};

struct Path_Information_t
{
    vector<double> x_values;
    vector<double> y_values;
};

/* Identify current lane using the input d value */
static Lane_e IdentifyLane(const double d_in);

/* Predict traffic information to know if another car is to the right side, to the left side or in
 * front of the ego vehicle */
static Traffic_Information_t Predict_TrafficInformation(const Sensor_Fusion_Data_t OtherVehicle,
                                                        const Localization_Data_t EgoVehicle);

/* Check the predictions about the current traffic situation and determine the upcoming behavior
   of the ego vehicle */
static void UpdateVehicleBehavior(Localization_Data_t& EgoVehicle,
                                  const Traffic_Information_t predictions);

/* Provides reference values used for trajectory generation */
template<typename T1, typename T2>
static Reference_Information_t GetReferenceValues(const Localization_Data_t EgoVehicle,
                                                  const T1 PreviousPath_x, const T2 PreviousPath_y);

/* Responsible for generating the trajectories of the path the ego vehicle is going to drive next */
template<typename T1, typename T2>
static Path_Information_t GenerateTrajectory(const Localization_Data_t EgoVehicle,
                                             const T1 PreviousPath_x, const T2 PreviousPath_y,
                                             const vector<double> Map_x, const vector<double> Map_y,
                                             const vector<double> Map_s);


static Lane_e IdentifyLane(const double d_in)
{
    Lane_e currentLane;

    if ((LEFT_LANE_BORDER_LEFT < d_in) && (d_in < LEFT_LANE_BORDER_RIGHT))
    {
        currentLane = LEFT_LANE;
    }
    else if ((LEFT_LANE_BORDER_RIGHT < d_in) && (d_in < RIGHT_LANE_BORDER_LEFT))
    {
        currentLane = MIDDLE_LANE;
    }
    else if ((RIGHT_LANE_BORDER_LEFT < d_in) && (d_in < RIGHT_LANE_BORDER_RIGHT))
    {
        currentLane = RIGHT_LANE;
    }
    else
    {
        currentLane = NO_LANE;
    }

    return currentLane;
}

static Traffic_Information_t Predict_TrafficInformation(Sensor_Fusion_Data_t OtherVehicle,
                                                        const Localization_Data_t EgoVehicle)
{
    Traffic_Information_t prediction ={false, false, false};
    Lane_e laneOtherVehicle;
    double speedOtherVehicle;

    /* Detect lane of other vehicle by using fused sensor data */
    laneOtherVehicle = IdentifyLane(OtherVehicle.d);

    /* Calculate vehicle speed */
    speedOtherVehicle = sqrt(OtherVehicle.vx * OtherVehicle.vx + OtherVehicle.vy * OtherVehicle.vy);

    OtherVehicle.s += ((double)EgoVehicle.previousPathSize * speedOtherVehicle * CYCLE_TIME);

    /* Compare detected lane of other vehicle with lane of ego vehicle */

    int laneOffset = (int)EgoVehicle.lane - (int)laneOtherVehicle;
    if ((EgoVehicle.lane != NO_LANE) && (laneOtherVehicle != NO_LANE))
    {
        /* Set the car ahead flag if the vehicles are on the same lane and the other vehcile is within 30m
         * ahead to the ego vehicle */
        if ((laneOffset == 0) && (OtherVehicle.s > EgoVehicle.s) &&
                ((OtherVehicle.s - EgoVehicle.s) < CRITICAL_DIST_TO_EGO_VEHICLE))
        {
            prediction.CarAhead_b = true;
        }
        /* Set the car right flag if the other vehicle is on a lane right to the lane of the ego vehicle
         *  and the gap between them is less than 30m */
        else if ((laneOffset == -1) && (abs(OtherVehicle.s - EgoVehicle.s) < CRITICAL_DIST_TO_EGO_VEHICLE))
        {
            prediction.CarRight_b = true;
        }
        /* Set the car left flag if the other vehicle is on a lane left to the lane of the ego vehicle
         *  and the gap between them is less than 30m */
        else if ((laneOffset == 1) && (abs(OtherVehicle.s - EgoVehicle.s) < CRITICAL_DIST_TO_EGO_VEHICLE))
        {
            prediction.CarLeft_b = true;
        }
        else
        {
            ;
        }
    }
    return prediction;
}

static void UpdateVehicleBehavior(Localization_Data_t& EgoVehicle,
                                  const Traffic_Information_t predictions)
{
    if (predictions.CarAhead_b) // If car is ahead, try to pass via right or left lane,
                                // if both not possible slow sown
    {
        if ((false == predictions.CarLeft_b) && (RIGHT_LANE == EgoVehicle.lane))
        {
            EgoVehicle.lane = MIDDLE_LANE;
        }
        else if ((false == predictions.CarLeft_b) && (MIDDLE_LANE == EgoVehicle.lane))
        {
            EgoVehicle.lane = LEFT_LANE;
        }
        else if ((false == predictions.CarRight_b) && (LEFT_LANE == EgoVehicle.lane))
        {
            EgoVehicle.lane = MIDDLE_LANE;
        }
        else if ((false == predictions.CarRight_b) && (MIDDLE_LANE == EgoVehicle.lane))
        {
            EgoVehicle.lane = RIGHT_LANE;
        }
        else
        {
            EgoVehicle.v_ref -= SPEED_DELTA;
        }
    }
    // Speed up within the range of the desired maximum speed limit if current lane is free
    else
    {
        if (((EgoVehicle.lane == LEFT_LANE) && (false == predictions.CarRight_b)) ||
                 ((EgoVehicle.lane == RIGHT_LANE) && (false == predictions.CarLeft_b)))
        {
            EgoVehicle.lane = MIDDLE_LANE;
        }
        if (EgoVehicle.v_ref < (MAX_SPEED - SPEED_DELTA))
        {
            EgoVehicle.v_ref += SPEED_DELTA;
        }
    }
}

template<typename T1, typename T2>
static Path_Information_t GenerateTrajectory(const Localization_Data_t EgoVehicle,
                                             const T1 PreviousPath_x, const T2 PreviousPath_y,
                                             const vector<double> Map_x, const vector<double> Map_y,
                                             const vector<double> Map_s)
{
    Path_Information_t tmp_trajectory;
    Reference_Information_t references;

    /* Final planned path, car visits every (x,y) point it recieves in the list every .02 seconds */
    Path_Information_t next_trajectory;

    /* Get reference values as the last two points from the previous path and put them as starting points
       in the temporary planned path */
    references = GetReferenceValues(EgoVehicle, PreviousPath_x, PreviousPath_y);

    tmp_trajectory.x_values.push_back(references.x_prev);
    tmp_trajectory.x_values.push_back(references.x);

    tmp_trajectory.y_values.push_back(references.y_prev);
    tmp_trajectory.y_values.push_back(references.y);

    vector<double> nextPoint;

    uint8_t currentLane_u8 = (uint8_t)EgoVehicle.lane;

    /* Append 30m evenly spaced points to the defined starting points */
    for (uint8_t i_u8 = 1; i_u8 < 4; i_u8++)
    {
        nextPoint = getXY(EgoVehicle.s + i_u8 * NEXT_POINTS_SPACE, currentLane_u8 * 4 + 2,
                                             Map_s, Map_x, Map_y);
        tmp_trajectory.x_values.push_back(nextPoint[0]);
        tmp_trajectory.y_values.push_back(nextPoint[1]);
    }

    /* Transform the points of the temporary planned path to the coordinates of the car */

    for (uint16_t i_u8 = 0U; i_u8 < tmp_trajectory.x_values.size(); i_u8++)
    {
        double shifted_x = tmp_trajectory.x_values[i_u8] - references.x;
        double shifted_y = tmp_trajectory.y_values[i_u8] - references.y;

        /* car reference angle shall be shifted to 0 degrees */

        tmp_trajectory.x_values[i_u8] = shifted_x * cos(0 - references.yaw) -
                shifted_y * sin(0 - references.yaw);
        tmp_trajectory.y_values[i_u8]  = shifted_x * sin(0 - references.yaw) +
                shifted_y * cos(0 - references.yaw);
    }

    /* Define and set points to the spline */

    tk::spline s;
    s.set_points(tmp_trajectory.x_values, tmp_trajectory.y_values);

    /* Make life easier by using points from the previous path, the vehicle has not passed yet */

    for (uint16_t i_u16 = 0U; i_u16 < EgoVehicle.previousPathSize; i_u16++)
    {
        next_trajectory.x_values.push_back(PreviousPath_x[i_u16]);
        next_trajectory.y_values.push_back(PreviousPath_y[i_u16]);
    }

    /* Define horizon for spline */

    double offset_x = 0;
    double horizon_x = 30.0;
    double horizon_y = s(horizon_x);
    double horizon_dist = sqrt(horizon_x * horizon_x + horizon_y * horizon_y);

    /* Fill missing points of the new planned path with points from the spline */

    for (uint16_t i_u16 = 1U; i_u16 < POINTS_PER_PATH - EgoVehicle.previousPathSize; i_u16++)
    {

        double N = horizon_dist / (CYCLE_TIME * EgoVehicle.v_ref / 2.24);
        double new_x = offset_x + horizon_x / N;
        double new_y = s(new_x);

        offset_x = new_x;

        double ref_x = new_x;
        double ref_y = new_y;

        /* Rotate back to global coordinate system */

        new_x = ref_x * cos(references.yaw) - ref_y * sin(references.yaw);
        new_y = ref_x * sin(references.yaw) + ref_y * cos(references.yaw);

        new_x += references.x;
        new_y += references.y;

        next_trajectory.x_values.push_back(new_x);
        next_trajectory.y_values.push_back(new_y);
    }

    return next_trajectory;
}

template<typename T1, typename T2>
static Reference_Information_t GetReferenceValues(const Localization_Data_t egoVehicle,
                                                  const T1 previousPath_x, const T2 previousPath_y)
{
    Reference_Information_t reference_values;

    if (egoVehicle.previousPathSize < 2)
    {
        reference_values.x = egoVehicle.x;
        reference_values.y = egoVehicle.y;
        reference_values.yaw = deg2rad(egoVehicle.yaw);

        reference_values.x_prev = egoVehicle.x - cos(egoVehicle.yaw);
        reference_values.y_prev = egoVehicle.y - sin(egoVehicle.yaw);
    }
    else
    {
        reference_values.x = previousPath_x[egoVehicle.previousPathSize - 1];
        reference_values.y = previousPath_y[egoVehicle.previousPathSize - 1];

        reference_values.x_prev = previousPath_x[egoVehicle.previousPathSize - 2];
        reference_values.y_prev = previousPath_y[egoVehicle.previousPathSize - 2];

        reference_values.yaw = atan2(reference_values.y - reference_values.y_prev,
                                     reference_values.x - reference_values.x_prev);
    }


    return reference_values;
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // Main car's localization Data
  Localization_Data_t egoVehicleData;

  egoVehicleData.v_ref = 0.0;
  egoVehicleData.lane = MIDDLE_LANE;

  h.onMessage([&egoVehicleData, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry")
        {
          // j[1] is the data JSON object
          


          egoVehicleData.x = j[1]["x"];
          egoVehicleData.y = j[1]["y"];
          egoVehicleData.s = j[1]["s"];
          egoVehicleData.d = j[1]["d"];
          egoVehicleData.yaw = j[1]["yaw"];
          egoVehicleData.v = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          egoVehicleData.previousPathSize = previous_path_x.size();

          /* Update the current s coordinate of the car */
          if (egoVehicleData.previousPathSize > 0) {
            egoVehicleData.s = end_path_s;
          }

          /* PREDICTION */

          Traffic_Information_t predictedTrafficInfo = {false, false, false};
          Traffic_Information_t tempPredictions = {false, false, false};
          Sensor_Fusion_Data_t sensor_data;

          for (uint16_t i_u16 = 0U; i_u16 < sensor_fusion.size(); i_u16++)
          {
              sensor_data.vx = sensor_fusion[i_u16][3];
              sensor_data.vy = sensor_fusion[i_u16][4];
              sensor_data.s = sensor_fusion[i_u16][5];
              sensor_data.d = sensor_fusion[i_u16][6];

              tempPredictions = Predict_TrafficInformation(sensor_data, egoVehicleData);

              predictedTrafficInfo.CarAhead_b |= tempPredictions.CarAhead_b;
              predictedTrafficInfo.CarRight_b |= tempPredictions.CarRight_b;
              predictedTrafficInfo.CarLeft_b |= tempPredictions.CarLeft_b;
          }


          /* BEHAVIORAL PLANNING */

          UpdateVehicleBehavior(egoVehicleData, predictedTrafficInfo);

          /* TRAJECTORY GENERATION */

          Path_Information_t nextPath;
          nextPath = GenerateTrajectory(egoVehicleData, previous_path_x, previous_path_y,
                                        map_waypoints_x, map_waypoints_y, map_waypoints_s);
          vector<double> next_x_vals = nextPath.x_values;
          vector<double> next_y_vals = nextPath.y_values;

          json msgJson;

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}
