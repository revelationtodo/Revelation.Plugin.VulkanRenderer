cmake_minimum_required(VERSION 3.14)

project(VulkanRenderer)

if (NOT IS_DIRECTORY $ENV{VULKAN_SDK})
    message(WARNING "Vulkan SDK not detected, if installed please set VULKAN_SDK env var or manually provide libraries")
else()
    message(STATUS "Found Vulkan SDK in $ENV{VULKAN_SDK}")
endif()

# Volk, VMA, glm and Slang are taken from the SDK
include_directories($ENV{VULKAN_SDK}/include)

OPTION(USE_D2D_WSI "Build the project using Direct to Display swapchain" OFF)
OPTION(USE_WAYLAND_WSI "Build the project using Wayland swapchain" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_library(Slang_LIBRARY NAMES slang HINTS "$ENV{VULKAN_SDK}/lib" REQUIRED)
find_package(ktx CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
elseif(LINUX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WAYLAND_KHR")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES")
if(MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

find_package(Qt6 COMPONENTS Widgets REQUIRED)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)

add_library(${PROJECT_NAME} SHARED
               dllmain.cpp
               VulkanRendererInterface.cpp
               VulkanRendererInterface.h

               VulkanAdapter.cpp
               VulkanAdapter.h
               VulkanRendererWidget.cpp
               VulkanRendererWidget.h

               Event/Events.h
               Event/EventQueue.cpp
               Event/EventQueue.h

               Parser/Parser.cpp
               Parser/Parser.h
)

if(MSVC)
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER Revelation)
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "$<1:${REVELATION_OUTPUT_DIR}/Debug/extensions/${PROJECT_NAME}>"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "$<1:${REVELATION_OUTPUT_DIR}/Release/extensions/${PROJECT_NAME}>"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "$<1:${REVELATION_OUTPUT_DIR}/RelWithDebInfo/extensions/${PROJECT_NAME}>"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "$<1:${REVELATION_OUTPUT_DIR}/Debug/extensions/${PROJECT_NAME}>"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "$<1:${REVELATION_OUTPUT_DIR}/Release/extensions/${PROJECT_NAME}>"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "$<1:${REVELATION_OUTPUT_DIR}/RelWithDebInfo/extensions/${PROJECT_NAME}>"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "$<1:${REVELATION_OUTPUT_DIR}/Debug/extensions/${PROJECT_NAME}>"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "$<1:${REVELATION_OUTPUT_DIR}/Release/extensions/${PROJECT_NAME}>"
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "$<1:${REVELATION_OUTPUT_DIR}/RelWithDebInfo/extensions/${PROJECT_NAME}>"
)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Od")
add_definitions(-DNOMINMAX)
endif()

add_definitions(-D_CRT_SECURE_NO_WARNINGS -DVK_NO_PROTOTYPES)

target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp KTX::ktx ${Slang_LIBRARY})

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets)
target_link_libraries(${PROJECT_NAME} PRIVATE FluUtils)
target_link_libraries(${PROJECT_NAME} PRIVATE FluControls)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../Include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../FluentUI/FluControls)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../FluentUI/FluUtils)

# resource copy
if(CMAKE_CONFIGURATION_TYPES)
    set(RENDERER_RES_DST ${REVELATION_OUTPUT_DIR}/$<CONFIG>/resources)
else()
    set(RENDERER_RES_DST ${REVELATION_OUTPUT_DIR}/${CMAKE_BUILD_TYPE}/resources)
endif()

set(REVELATION_RES_DST ${PROJECT_BINARY_DIR}/../Revelation/resources)

add_custom_target(RendererResourceCopy)

set(RENDERER_RESOURCE_DIRS shader skybox)
foreach(dir IN LISTS RENDERER_RESOURCE_DIRS)
    add_custom_command(TARGET RendererResourceCopy PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory
                ${RENDERER_RES_DST}/${dir}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/resources/${dir}
                ${RENDERER_RES_DST}/${dir}

        COMMAND ${CMAKE_COMMAND} -E remove_directory
                ${REVELATION_RES_DST}/${dir}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/resources/${dir}
                ${REVELATION_RES_DST}/${dir}
    )
endforeach()

set_property(TARGET RendererResourceCopy PROPERTY FOLDER ResourceCopy)
add_dependencies(VulkanRenderer RendererResourceCopy)