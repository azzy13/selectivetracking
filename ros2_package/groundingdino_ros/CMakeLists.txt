cmake_minimum_required(VERSION 3.8)
project(groundingdino_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GroundingDINOTrack.msg"
  "msg/GroundingDINOTrackArray.msg"
  DEPENDENCIES std_msgs
)

# Install Python executables
install(PROGRAMS
  scripts/groundingdino_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules directly (manual installation to avoid conflicts)
install(FILES
  ${PROJECT_NAME}/groundingdino_node.py
  ${PROJECT_NAME}/mission_parser.py
  ${PROJECT_NAME}/geometry.py
  ${PROJECT_NAME}/perception_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
