cmake_minimum_required(VERSION 2.8)

ADD_DEFINITIONS(
    -std=c++11 
)


project( readData )
find_package( OpenCV REQUIRED )

#find_package(PCL 1.3 REQUIRED COMPONENTS common io)

#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})


add_executable( readData  readData.cpp )
target_link_libraries( readData ${OpenCV_LIBS})

#${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} 
