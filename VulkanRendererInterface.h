#pragma once
#include "IExtensionInterface.h"
#include "IRevelationInterface.h"

class VulkanRendererWidget;

class VulkanRendererInterface : public IExtensionInterface
{
  public:
    VulkanRendererInterface(IRevelationInterface* intf);
    ~VulkanRendererInterface();

    virtual void Initialize() override;
    virtual void Uninitialize() override;

    virtual void HandleBroadcast(BroadcastType broadcastType, const std::any& param /* = std::any() */) override;

  private:
    void AddNavigationView();
    void AddSettingsItem();

  private:
    IRevelationInterface* m_interface = nullptr;

    VulkanRendererWidget* m_rendererWidget = nullptr;
};