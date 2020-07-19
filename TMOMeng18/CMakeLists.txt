CMAKE_MINIMUM_REQUIRED(VERSION 2.4.8)
SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_INCLUDE_CURRENT_DIR TRUE)

IF (UNIX AND NOT LINUX_SET)
    ADD_DEFINITIONS(-D LINUX)
ENDIF (UNIX AND NOT LINUX_SET)



INCLUDE_DIRECTORIES(
        ../tmolib/
        .
        ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/
        ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/tensorflow/include
#        /mnt/DATA/FIT/tensorflow/bazel-bin/tensorflow
#        /mnt/DATA/FIT/tensorflow/bazel-bin/tensorflow/include

)
LINK_LIBRARIES(
        ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/tensorflow/libtensorflow_framework.so
        ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/tensorflow/libtensorflow_cc.so
#        /mnt/DATA/FIT/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so
#        /mnt/DATA/FIT/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
#add_executable(main ${main_SRC} main.cpp VGG19.h ImageLoader.h InvertibleGrayscale.h InvertibleGrayscale.cpp)
#target_link_libraries( main ${OpenCV_LIBS} )
SET(TMO_SOURCES
        TMOMeng18.cpp
        TMOPlugin.cpp
        VGG19.cpp
        ImageLoader.cpp
        InvertibleGrayscale.cpp
        )


SET(TMOMENG_HEADERS
        TMOMeng18.h
        TMOPlugin.h
        VGG19.h
        ImageLoader.h
        InvertibleGrayscale.h
        )

add_library(meng18 SHARED ${TMOMENG_HEADERS} ${TMO_SOURCES})
#add_dependencies(meng18 tensorflow)

add_custom_command(TARGET meng18 POST_BUILD
        COMMAND cp -f libmeng18.so ../TMOCmd/meng18.tml
        COMMAND cp -f libmeng18.so ../TMOgui/meng18.tml
        COMMAND cp -f libmeng18.so ../meng18.tml
        #	COMMAND rm -f libmeng18.so
        )


