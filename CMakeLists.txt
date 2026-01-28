cmake_minimum_required(VERSION 3.14)

project(VulkanRenderer)

if (NOT IS_DIRECTORY $ENV{VULKAN_SDK})
    message(WARNING "Vulkan SDK not detected, if installed please set VULKAN_SDK env var or manually provide libraries")
else()
    message(STATUS "Found Vulkan SDK in $ENV{VULKAN_SDK}")
endif()

# Volk, VMA, glm and Slang are taken from the SDK
include_directories($ENV{VULKAN_SDK}/include)
# These are not part of the SDK
include_directories(external/tinyobj)
include_directories(external/ktx/include)
include_directories(external/ktx/other_include)

OPTION(USE_D2D_WSI "Build the project using Direct to Display swapchain" OFF)
OPTION(USE_WAYLAND_WSI "Build the project using Wayland swapchain" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(KTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/ktx)
set(KTX_SOURCES
    ${KTX_DIR}/lib/texture.c
    ${KTX_DIR}/lib/hashlist.c
    ${KTX_DIR}/lib/checkheader.c
    ${KTX_DIR}/lib/swap.c
    ${KTX_DIR}/lib/memstream.c
    ${KTX_DIR}/lib/filestream.c
    ${KTX_DIR}/lib/vkloader.c)

add_library(ktx STATIC ${KTX_SOURCES})
if(MSVC)
    set_property(TARGET ktx PROPERTY FOLDER Dependencies/Ktx)
endif()

find_library(Slang_LIBRARY NAMES slang HINTS "$ENV{VULKAN_SDK}/lib" REQUIRED)

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

               Parser/IParser.h
               Parser/ParserManager.cpp
               Parser/ParserManager.h
               Parser/WavefrontObjParser.cpp
               Parser/WavefrontObjParser.h
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

target_link_libraries(${PROJECT_NAME} PRIVATE ktx ${Slang_LIBRARY})

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets)
target_link_libraries(${PROJECT_NAME} PRIVATE FluUtils)
target_link_libraries(${PROJECT_NAME} PRIVATE FluControls)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../Include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../FluentUI/FluControls)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../FluentUI/FluUtils)


if(CMAKE_CONFIGURATION_TYPES)
    add_custom_target(ShaderResourceCopy
        COMMAND ${CMAKE_COMMAND} -E remove_directory
                ${REVELATION_OUTPUT_DIR}/$<CONFIG>/resources/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/assets
                ${REVELATION_OUTPUT_DIR}/$<CONFIG>/resources/shaders

        COMMAND ${CMAKE_COMMAND} -E remove_directory 
                ${PROJECT_BINARY_DIR}/../Revelation/resources/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/assets
                ${PROJECT_BINARY_DIR}/../Revelation/resources/shaders
    )

    message(${PROJECT_BINARY_DIR}/Revelation/resources/shaders)
else()
    add_custom_target(ShaderResourceCopy
        COMMAND ${CMAKE_COMMAND} -E remove_directory
                ${REVELATION_OUTPUT_DIR}/${CMAKE_BUILD_TYPE}/resources/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/assets
                ${REVELATION_OUTPUT_DIR}/${CMAKE_BUILD_TYPE}/resources/shaders

        COMMAND ${CMAKE_COMMAND} -E remove_directory 
                ${PROJECT_BINARY_DIR}/../Revelation/resources/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/assets
                ${PROJECT_BINARY_DIR}/../Revelation/resources/shaders
    )
endif()

set_property(TARGET ShaderResourceCopy PROPERTY FOLDER ResourceCopy)
