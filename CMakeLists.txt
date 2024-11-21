cmake_minimum_required(VERSION 3.1)

project(f1tenth_simulator)
add_compile_options(-g)
#set vcpkg toolchain file in order to import xtensor package via vcpkg package manager.
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()
set(xtensor_DIR "$ENV{VCPKG_ROOT}/installed/arm64-linux/share/xtensor")
set(xtl_DIR "$ENV{VCPKG_ROOT}/installed/arm64-linux/share/xtl")
set(nlohmann_json_DIR "$ENV{VCPKG_ROOT}/installed/arm64-linux/share/nlohmann_json")
#set(nlohmann-json_DIR "$ENV{VCPKG_ROOT}/installed/arm64-linux/share/nlohmann-json")

set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g") 
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Eigen3 REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(realsense2 CONFIG REQUIRED)
######################################
# Compile the library >
######################################

# Add includes
include_directories(include ${EIGEN_3_INCLUDE_DIRS} ${xtensor_INCLUDE_DIRS})
file(GLOB SRC_FILES src/*.cpp)
add_library(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${LIBS} xtensor realsense2::realsense2)

set(LIBS ${LIBS} ${PROJECT_NAME})

# Install the library to CMAKE_INSTALL_PREFIX
# which defaults to /usr/local
install(TARGETS ${PROJECT_NAME}
        DESTINATION lib)
install(DIRECTORY include/${PROJECT_NAME}
        DESTINATION include)

######################################
# < End compile the library
######################################

######################################
# ROS >
######################################

# Only run this code in the catkin environment
if(DEFINED CATKIN_DEVEL_PREFIX)
  find_package(catkin REQUIRED COMPONENTS
    roslib
    roscpp
    tf2
    tf2_ros
    tf2_geometry_msgs
    ackermann_msgs
    nav_msgs
    sensor_msgs
    geometry_msgs
    tf
    interactive_markers
    visualization_msgs
    cv_bridge
    image_transport

    std_msgs
    message_generation)
  
add_message_files(
    FILES
    YoloData.msg
  )


  generate_messages(
    DEPENDENCIES
    std_msgs
    nav_msgs
    sensor_msgs
    geometry_msgs
    )




  # Include catkin_libraries
  catkin_package(
      INCLUDE_DIRS include
      LIBRARIES ${PROJECT_NAME}
      CATKIN_DEPENDS 
      message_runtime
      cv_bridge
      image_transport    

      		     
  )
  set(LIBS ${LIBS} ${catkin_LIBRARIES})
  include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
    /usr/local/include/QuadProg++
    /usr/local/include/nlopt
  )



  

  # Add the nodes
  file(GLOB NODE_SRC_FILES node/*.cpp)
  foreach(_node_file ${NODE_SRC_FILES})
      get_filename_component(_node_name ${_node_file} NAME_WE)
      add_executable(${_node_name} ${_node_file})
      target_link_libraries(${_node_name} ${LIBS})
      add_dependencies(${_node_name} f1tenth_simulator_generate_messages_cpp)
  endforeach()

  find_library(QUADPROGPP_LIBRARIES NAMES quadprog)
  find_path(QUADPROGPP_INCLUDE_DIR NAMES QuadProg++.hh)
  target_link_libraries(navigation ${QUADPROGPP_LIBRARIES})

  find_library(NLOPT_LIBRARIES NAMES nlopt)
  find_path(NLOPT_INCLUDE_DIR NAMES nlopt.hpp)
  target_link_libraries(navigation ${NLOPT_LIBRARIES})



  # Install the library
  install(TARGETS ${PROJECT_NAME}
      ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
      LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
      RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
  install(DIRECTORY include/${PROJECT_NAME}/
      DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  )
endif()

######################################
# < End ROS
######################################