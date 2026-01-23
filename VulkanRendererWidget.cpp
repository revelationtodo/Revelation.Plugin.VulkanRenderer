#include "VulkanRendererWidget.h"
#include "VulkanAdapter.h"

VulkanRendererWidget::VulkanRendererWidget(IRevelationInterface* intf, QWidget* parent /*= nullptr*/)
    : QWidget(parent)
{
    Initialize();
}

VulkanRendererWidget::~VulkanRendererWidget()
{
}


void VulkanRendererWidget::Initialize()
{
    m_adapter = new VulkanAdapter(this);
    m_adapter->Initialize();

    InitWidget();
    InitSignalSlots();
}

void VulkanRendererWidget::InitWidget()
{
}

void VulkanRendererWidget::InitSignalSlots()
{
}
