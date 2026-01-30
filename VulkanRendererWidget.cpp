#include "VulkanRendererWidget.h"
#include "VulkanAdapter.h"
#include <QGridLayout>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QDir>

VulkanRendererWidget::VulkanRendererWidget()
{
    Initialize();
}

VulkanRendererWidget::~VulkanRendererWidget()
{
    delete m_adapter;
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

std::optional<Event> VulkanRendererWidget::PollEvent()
{
    return dynamic_cast<VulkanRendererWidgetWrapper*>(m_wrapper)->PollEvent();
}

void VulkanRendererWidget::showEvent(QShowEvent* event)
{
    if (!m_frameTimer.isActive())
    {
        m_frameTimer.start();
    }
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

VulkanRendererWidgetWrapper::VulkanRendererWidgetWrapper(QWidget* parent /*= nullptr*/)
    : QWidget(parent)
{
    Initialize();
}

VulkanRendererWidgetWrapper::~VulkanRendererWidgetWrapper()
{
    delete m_rendererWidget;
}

std::optional<Event> VulkanRendererWidgetWrapper::PollEvent()
{
    if (m_eventQueue.Empty())
    {
        return std::nullopt;
    }

    return m_eventQueue.Pop();
}

bool VulkanRendererWidgetWrapper::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == m_rendererWidget)
    {
        switch (event->type())
        {
            case QEvent::MouseButtonPress:
                mousePressEvent(static_cast<QMouseEvent*>(event));
                return true;
            case QEvent::MouseButtonRelease:
                mouseReleaseEvent(static_cast<QMouseEvent*>(event));
                return true;
            case QEvent::MouseButtonDblClick:
                mouseDoubleClickEvent(static_cast<QMouseEvent*>(event));
                return true;
            case QEvent::MouseMove:
                mouseMoveEvent(static_cast<QMouseEvent*>(event));
                return true;
            case QEvent::Wheel:
                wheelEvent(static_cast<QWheelEvent*>(event));
                return true;
            default:
                break;
        }
    }
    return QWidget::eventFilter(watched, event);
}

void VulkanRendererWidgetWrapper::resizeEvent(QResizeEvent* event)
{
    Event e{.type = EventType::ResizeEvent,
            .data = ResizeEventData{}};
    m_eventQueue.Push(std::move(e));
}

void VulkanRendererWidgetWrapper::dragEnterEvent(QDragEnterEvent* event)
{
    const QMimeData* mime = event->mimeData();
    if (nullptr != mime && mime->hasUrls())
    {
        event->acceptProposedAction();
        return;
    }
    event->ignore();
}

void VulkanRendererWidgetWrapper::dragMoveEvent(QDragMoveEvent* event)
{
    const QMimeData* mime = event->mimeData();
    if (nullptr != mime && mime->hasUrls())
    {
        event->acceptProposedAction();
        return;
    }
    event->ignore();
}

void VulkanRendererWidgetWrapper::dropEvent(QDropEvent* event)
{
    const QMimeData* mime = event->mimeData();
    if (nullptr == mime || !mime->hasUrls())
    {
        event->ignore();
        return;
    }

    const QList<QUrl>        urls = mime->urls();
    std::vector<std::string> paths;
    paths.reserve(urls.size());
    for (const QUrl& url : urls)
    {
        const QString path = url.toLocalFile();
        if (!path.isEmpty())
        {
            paths.push_back(path.toStdString());
        }
    }

    if (!paths.empty())
    {
        Event e{.type = EventType::DropEvent,
                .data = DropEventData{.file = paths[0]}};
        m_eventQueue.Push(std::move(e));

        event->acceptProposedAction();
        return;
    }

    event->ignore();
}

void VulkanRendererWidgetWrapper::mousePressEvent(QMouseEvent* event)
{
    MouseBtnType btnType = MouseBtnType::None;
    if (event->button() == Qt::LeftButton)
    {
        m_leftBtnPressed = true;
        btnType          = MouseBtnType::Left;
    }
    else if (event->button() == Qt::RightButton)
    {
        m_rightBtnPressed = true;
        btnType           = MouseBtnType::Right;
    }

    Event e{.type = EventType::MouseEvent,
            .data = MouseEventData{.event = MouseEventType::Press,
                                   .btn   = btnType}};
    m_eventQueue.Push(std::move(e));
}

void VulkanRendererWidgetWrapper::mouseReleaseEvent(QMouseEvent* event)
{
    MouseBtnType btnType = MouseBtnType::None;
    if (event->button() == Qt::LeftButton)
    {
        btnType = MouseBtnType::Left;
    }
    else if (event->button() == Qt::RightButton)
    {
        btnType = MouseBtnType::Right;
    }

    Event e{.type = EventType::MouseEvent,
            .data = MouseEventData{.event = MouseEventType::Release,
                                   .btn   = btnType}};
    m_eventQueue.Push(std::move(e));

    if (event->button() == Qt::LeftButton)
    {
        m_leftBtnPressed = false;
    }
    else if (event->button() == Qt::RightButton)
    {
        m_rightBtnPressed = false;
    }
}

void VulkanRendererWidgetWrapper::mouseDoubleClickEvent(QMouseEvent* event)
{
    MouseBtnType btnType = MouseBtnType::None;
    if (event->button() == Qt::LeftButton)
    {
        btnType = MouseBtnType::Left;
    }
    else if (event->button() == Qt::RightButton)
    {
        btnType = MouseBtnType::Right;
    }

    Event e{.type = EventType::MouseEvent,
            .data = MouseEventData{.event = MouseEventType::DoubleClick,
                                   .btn   = btnType}};
    m_eventQueue.Push(std::move(e));
}

void VulkanRendererWidgetWrapper::mouseMoveEvent(QMouseEvent* event)
{
    QPoint pos   = event->pos();
    QPoint delta = pos - m_lastPoint;
    m_lastPoint  = pos;

    Event e{.type = EventType::MouseEvent,
            .data = MouseEventData{.event           = MouseEventType::Move,
                                   .deltaX          = delta.x(),
                                   .deltaY          = delta.y(),
                                   .leftBtnPressed  = m_leftBtnPressed,
                                   .rightBtnPressed = m_rightBtnPressed}};
    m_eventQueue.Push(std::move(e));
}

void VulkanRendererWidgetWrapper::wheelEvent(QWheelEvent* event)
{
    QPoint delta = event->angleDelta();
    Event  e{.type = EventType::MouseEvent,
             .data = MouseEventData{.event  = MouseEventType::Wheel,
                                    .deltaX = delta.x(),
                                    .deltaY = delta.y()}};
    m_eventQueue.Push(std::move(e));
}

void VulkanRendererWidgetWrapper::Initialize()
{
    InitWidget();
    InitSignalSlots();
}

void VulkanRendererWidgetWrapper::InitWidget()
{
    m_rendererWidget = new VulkanRendererWidget;
    m_rendererWidget->SetWrapper(this);
    m_rendererWidget->installEventFilter(this);

    QWidget* container = QWidget::createWindowContainer(m_rendererWidget, nullptr);
    container->setAcceptDrops(true);
    container->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    QGridLayout* layout = new QGridLayout(this);
    layout->setSpacing(0);
    layout->setContentsMargins(8, 38, 8, 8);
    layout->addWidget(container);

    setAcceptDrops(true);
}

void VulkanRendererWidgetWrapper::InitSignalSlots()
{
}
