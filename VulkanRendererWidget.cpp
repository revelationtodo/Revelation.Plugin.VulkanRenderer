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

bool VulkanRendererWidget::IsResized()
{
    bool resized = m_resized;
    m_resized    = false;
    return resized;
}

void VulkanRendererWidget::LoadModel(const std::string& filePath)
{
    std::thread loadThread([this, filePath]() {
        Model model;
        if (m_parserManager->Parse(filePath, model))
        {
            m_adapter->LoadModel(model);
        }
    });
    loadThread.detach();
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

    m_parserManager = std::make_unique<ParserManager>();

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

VulkanRendererWidgetWrapper::VulkanRendererWidgetWrapper(QWidget* parent /*= nullptr*/)
    : QWidget(parent)
{
    Initialize();
}

VulkanRendererWidgetWrapper::~VulkanRendererWidgetWrapper()
{
    delete m_rendererWidget;
}

bool VulkanRendererWidgetWrapper::eventFilter(QObject* watched, QEvent* event)
{
    switch (event->type())
    {
        case QEvent::DragEnter:
            dragEnterEvent(static_cast<QDragEnterEvent*>(event));
            return true;
        case QEvent::DragMove:
            dragMoveEvent(static_cast<QDragMoveEvent*>(event));
            return true;
        case QEvent::Drop:
            dropEvent(static_cast<QDropEvent*>(event));
            return true;
        default:
            break;
    }
    return QWidget::eventFilter(watched, event);
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
        m_rendererWidget->LoadModel(paths[0]);
        event->acceptProposedAction();
        return;
    }

    event->ignore();
}

void VulkanRendererWidgetWrapper::Initialize()
{
    InitWidget();
    InitSignalSlots();
}

void VulkanRendererWidgetWrapper::InitWidget()
{
    m_rendererWidget   = new VulkanRendererWidget;
    QWidget* container = QWidget::createWindowContainer(m_rendererWidget, nullptr);
    container->setAcceptDrops(true);
    container->installEventFilter(this);

    m_rendererWidget->SetWrapper(this);
    QGridLayout* layout = new QGridLayout(this);
    layout->setSpacing(0);
    layout->setContentsMargins(8, 38, 8, 8);
    layout->addWidget(container);

    setAcceptDrops(true);
}

void VulkanRendererWidgetWrapper::InitSignalSlots()
{
}
