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
    InitWidget();
    InitSignalSlots();

    m_adapter = new VulkanAdapter(this);
    m_adapter->Initialize();

    m_clock.start();
    m_lastNs = m_clock.nsecsElapsed();

    m_frameTimer.setTimerType(Qt::PreciseTimer);
    m_frameTimer.setInterval(16);

    connect(&m_frameTimer, &QTimer::timeout, this, &VulkanRendererWidget::TriggerTick);

    m_frameTimer.start();
}

void VulkanRendererWidget::InitWidget()
{
    setMouseTracking(true);
}

void VulkanRendererWidget::InitSignalSlots()
{
}

void VulkanRendererWidget::TriggerTick()
{
    if (m_adapter->IsReady())
    {
        const qint64 nowNs   = m_clock.nsecsElapsed();
        qint64       deltaNs = nowNs - m_lastNs;
        m_lastNs             = nowNs;

        double deltaSeconds = deltaNs / 1e9;
        deltaSeconds        = std::clamp(deltaSeconds, 0.0, 0.1);
        m_adapter->Tick(deltaSeconds);
    }
}
