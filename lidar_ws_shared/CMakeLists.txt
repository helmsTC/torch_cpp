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
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# ============================================================================
# Shared Memory Version (No LibTorch needed)
# ============================================================================

# Create the shared memory component library
add_library(lidar_processor_shm_component SHARED
  src/lidar_shm.cpp
)

target_include_directories(lidar_processor_shm_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

target_compile_definitions(lidar_processor_shm_component
  PRIVATE "LIDAR_PROCESSOR_BUILDING_DLL"
)

ament_target_dependencies(lidar_processor_shm_component
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  visualization_msgs
  pcl_conversions
)

target_link_libraries(lidar_processor_shm_component
  ${PCL_LIBRARIES}
  pthread  # For threading
  rt       # For shared memory
)

# Register the component
rclcpp_components_register_node(
  lidar_processor_shm_component
  PLUGIN "lidar_processing::LidarProcessorSHM"
  EXECUTABLE lidar_processor_shm_node
)

# ============================================================================
# Optional: Original LibTorch Version (if you still want to keep it)
# ============================================================================

# Uncomment this section if you want to build the original LibTorch version
# set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
# find_package(Torch)
# 
# if(Torch_FOUND)
#   add_library(lidar_processor_component SHARED
#     src/lidar.cpp
#   )
#   
#   target_include_directories(lidar_processor_component PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>
#     ${TORCH_INCLUDE_DIRS}
#     ${PCL_INCLUDE_DIRS}
#   )
#   
#   ament_target_dependencies(lidar_processor_component
#     rclcpp
#     rclcpp_components
#     sensor_msgs
#     std_msgs
#     visualization_msgs
#     pcl_conversions
#   )
#   
#   target_link_libraries(lidar_processor_component
#     ${TORCH_LIBRARIES}
#     ${PCL_LIBRARIES}
#   )
#   
#   rclcpp_components_register_node(
#     lidar_processor_component
#     PLUGIN "lidar_processing::LidarProcessor"
#     EXECUTABLE lidar_processor_node
#   )
#   
#   install(TARGETS lidar_processor_component
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION lib/${PROJECT_NAME}
#   )
# else()
#   message(WARNING "LibTorch not found. Only shared memory version will be built.")
# endif()

# ============================================================================
# Installation
# ============================================================================

# Install the shared memory library
install(TARGETS lidar_processor_shm_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  scripts/maskpls_inference_server.py
#  scripts/prepare_model.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install the Python converter scripts
#install(PROGRAMS
#  ckpt_to_pt/original_pt_converter.py
#  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
