#include "VulkanRendererWidget.h"
#include "VulkanAdapter.h"

VulkanRendererWidget::VulkanRendererWidget(IRevelationInterface* intf)
{
    Initialize();
}

VulkanRendererWidget::~VulkanRendererWidget()
{
}

void VulkanRendererWidget::SetWrapper(QWidget* wrapper)
{
    m_wrapper = wrapper;
}

uint32_t VulkanRendererWidget::GetWidthPix()
{
    if (nullptr == m_wrapper)
    {
        QSize logical = size();
        qreal dpr     = devicePixelRatio();
        return uint32_t(logical.width() * dpr);
    }

    QSize logical = m_wrapper->size();
    qreal dpr     = m_wrapper->devicePixelRatioF();
    return uint32_t(logical.width() * dpr);
}

uint32_t VulkanRendererWidget::GetHeightPix()
{
    if (nullptr == m_wrapper)
    {
        QSize logical = size();
        qreal dpr     = devicePixelRatio();
        return uint32_t(logical.height() * dpr);
    }

    QSize logical = m_wrapper->size();
    qreal dpr     = m_wrapper->devicePixelRatio();
    return uint32_t(logical.height() * dpr);
}

bool VulkanRendererWidget::IsResized()
{
    bool resized = m_resized;
    m_resized    = false;
    return resized;
}

void VulkanRendererWidget::resizeEvent(QResizeEvent* event)
{
    m_resized = true;
    return QWindow::resizeEvent(event);
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
