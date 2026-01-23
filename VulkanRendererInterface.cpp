#include "VulkanRendererInterface.h"
#include "IRevelationInterface.h"
#include "ICommonWidgetInterface.h"
#include "VulkanRendererWidget.h"
#include "FluThemeUtils.h"
#include "FluDef.h"

VulkanRendererInterface::VulkanRendererInterface(IRevelationInterface* intf)
    : m_interface(intf)
{
}

VulkanRendererInterface::~VulkanRendererInterface()
{
    delete m_rendererWidget;
}

void VulkanRendererInterface::Initialize()
{
}

void VulkanRendererInterface::Uninitialize()
{
}

void VulkanRendererInterface::HandleBroadcast(BroadcastType broadcastType, const std::any& param /* = std::any() */)
{
    if (broadcastType == BroadcastType::CollectNavigationView)
    {
        AddNavigationView();
    }
    else if (broadcastType == BroadcastType::CollectSettingsItem)
    {
        AddSettingsItem();
    }
    else if (broadcastType == BroadcastType::ChangeTheme)
    {
        FluTheme theme = std::any_cast<FluTheme>(param);
        FluThemeUtils::getUtils()->setTheme(theme);
    }
}

void VulkanRendererInterface::AddNavigationView()
{
    auto commonWidgetIntf = m_interface->GetCommonWidgetInterface();
    if (nullptr != commonWidgetIntf)
    {
        m_rendererWidget = new VulkanRendererWidget(m_interface);
        commonWidgetIntf->AddStackedWidget(m_rendererWidget, QObject::tr("VulkanRenderer"), FluAwesomeType::EyeGaze, Qt::AlignCenter);
    }
}

void VulkanRendererInterface::AddSettingsItem()
{
}
