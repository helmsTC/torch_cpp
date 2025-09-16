cmake_minimum_required(VERSION 3.8)
project(lidar_processor)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Find LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(
  include
  ${TORCH_INCLUDE_DIRS}
)

# Create the component library
add_library(lidar_processor_component SHARED
  src/lidar.cpp
)

target_include_directories(lidar_processor_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_definitions(lidar_processor_component
  PRIVATE "LIDAR_PROCESSOR_BUILDING_DLL"
)

ament_target_dependencies(lidar_processor_component
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
)

target_link_libraries(lidar_processor_component
  ${TORCH_LIBRARIES}
)

# Register the component
rclcpp_components_register_node(
  lidar_processor_component
  PLUGIN "lidar_processing::LidarProcessor"
  EXECUTABLE lidar_processor_node
)

# Install the library
install(TARGETS lidar_processor_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Install launch files if any
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

# Install config files if any
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()