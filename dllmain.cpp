#include "VulkanRendererInterface.h"

DLL_EXPORT IExtensionInterface* ExtensionEntrance(IRevelationInterface* intf)
{
    return new VulkanRendererInterface(intf);
}
