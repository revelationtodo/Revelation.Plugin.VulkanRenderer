1. # Vulkan Renderer

   A lightweight Vulkan-based renderer plugin for displaying 3D models inside **Revelation TODO**.

   ---

   ## Overview

   **Vulkan Renderer** is a plugin designed for the Revelation ecosystem, providing real-time rendering of 3D models using the Vulkan API.  
   It supports drag-and-drop model loading and integrates seamlessly with the Revelation plugin system.

   ---

   ## Usage

   1. Download **Revelation TODO** from the [Releases](https://github.com/revelationtodo/Revelation/releases) page.
   2. Download the **Vulkan Renderer** plugin from this repository’s release page.
   3. Extract the downloaded `.zip` file into the Revelation installation directory.
   4. Launch `Revelation.exe` and open the **Vulkan Renderer** plugin page.
   5. Drag any supported 3D model file into the renderer window.
   6. The model will be loaded and rendered once the import process is complete.

   ---

   ## Vulkan Renderer

   一个基于 **Vulkan** 的轻量级 3D 模型渲染插件，用于在 **Revelation TODO** 中实时显示 3D 模型。

   ---

   ## 使用说明

   1. 从 [Releases](https://github.com/revelationtodo/Revelation/releases) 页面下载 **Revelation TODO** 主程序。
   2. 从本仓库的发布页面下载 **Vulkan Renderer** 插件。
   3. 将下载的 `.zip` 文件解压到 Revelation 的安装目录中。
   4. 启动 `Revelation.exe`，并切换到 **Vulkan Renderer** 插件页面。
   5. 将任意支持格式的 3D 模型文件拖拽到渲染窗口中。
   6. 模型加载完成后，将自动显示并渲染在窗口中。

   ---

   ## Build Instructions

   ### Prerequisites

   1. **Install Vulkan SDK**

      Download and install the Vulkan SDK from the official website:  
      https://vulkan.lunarg.com/

      Ensure the following components are available:

      - Vulkan SDK
      - `volk`
      - `glm`
      - Vulkan Memory Allocator (`vma`)

      Make sure the environment variable `VULKAN_SDK` is properly set.

   ---

   ### Build Steps

   1. **Clone the Revelation repository**

      ```bash
       git clone --recursive https://github.com/revelationtodo/Revelation.git
      ```

   2. **Add the Vulkan Renderer plugin**
      Navigate to the SourceCode directory inside the Revelation project, then clone this repository:

      ```bash
      cd Revelation/SourceCode
      git clone https://github.com/revelationtodo/Revelation.Plugin.VulkanRenderer.git VulkanRenderer
      ```

   3. **Merge vcpkg dependencies**
      Some dependencies defined in:

      ```bash
      VulkanRenderer/vcpkg.json
      ```

      may not yet exist in the main project configuration.
      Please manually copy any **missing dependency entries** into: 

      ```bash
      Revelation/SourceCode/vcpkg.json
      ```

   4. **Build Revelation**
      Follow the build instructions provided in the main Revelation repository:
      https://github.com/revelationtodo/Revelation

   5. Build and Run

      After completing the engine setup:

      - Build the project
      - Launch Revelation.exe
      - Switch to renderer page
      - Verify that the Vulkan Renderer plugin is correctly loaded
