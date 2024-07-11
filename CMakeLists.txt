cmake_minimum_required(VERSION 3.19)

project(RVV VERSION 1.0)

include(GNUInstallDirs)
include(CTest)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(IS_TOPLEVEL_PROJECT TRUE)
else()
  set(IS_TOPLEVEL_PROJECT FALSE)
endif()

option(RVV_BUILD_TESTING "Build and run RVV tests " ${IS_TOPLEVEL_PROJECT})
option(RVV_INSTALL_LIBRARY
       "Enable installing of RVV library into default locations"
       ${IS_TOPLEVEL_PROJECT})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# if (NOT RVV_LENGTH)
#   set(RVV_LENGTH 128)
# endif()
# message(STATUS "Using rvv length : ${RVV_LENGTH}")

add_library(rvv INTERFACE)
add_library(RVV::rvv ALIAS rvv)

target_include_directories(
  rvv INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(rvv INTERFACE cxx_std_20)
# target_compile_definitions(rvv INTERFACE RVV_LEN=${RVV_LENGTH})
target_compile_options(rvv INTERFACE "-march=rv64gcv" "-mrvv-vector-bits=zvl")

if(BUILD_TESTING AND RVV_BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(RVV_INSTALL_LIBRARY)
  install(
    TARGETS rvv
    EXPORT RVV_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "RVVConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/RVVConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/RVVConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rvv/cmake)

  install(
    EXPORT RVV_Targets
    FILE RVVTargets.cmake
    NAMESPACE RVV::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rvv/cmake)

  install(FILES "${PROJECT_BINARY_DIR}/RVVConfig.cmake"
                "${PROJECT_BINARY_DIR}/RVVConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rvv/cmake)

  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/rvv
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
