cmake_minimum_required(VERSION 2.8)

project( stereo-vo)

find_package( OpenCV 4.0 REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(GTSAM REQUIRED) # Uses installed package

# include_directories( ${GTSAM_INCLUDE_DIR} )
include_directories( ${OpenCV_INCLUDE_DIRS} )


file(GLOB viso
    "src/*.hpp"
    "src/*.cpp"
)

add_executable( vo ${viso} )

target_link_libraries( vo ${OpenCV_LIBS} )
target_link_libraries( vo Eigen3::Eigen)
# target_link_libraries( vo gtsam)