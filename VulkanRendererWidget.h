#pragma once
#include <QWindow>
#include <QWidget>
#include <QTimer>
#include <QElapsedTimer>
#include <vector>
#include <filesystem>
#include <optional>
#include "Event/EventQueue.h"

class IRevelationInterface;
class VulkanAdapter;

class VulkanRendererWidget : public QWindow
{
    Q_OBJECT

  public:
    VulkanRendererWidget();
    ~VulkanRendererWidget();

    void SetWrapper(QWidget* wrapper);

    uint32_t GetWidthPix();
    uint32_t GetHeightPix();

    std::optional<Event> PollEvent();

  protected:
    void showEvent(QShowEvent* event) override;

  private:
    void Initialize();
    void InitWidget();
    void InitSignalSlots();

  private slots:
    void TriggerTick();

  private:
    QWidget* m_wrapper = nullptr;

    VulkanAdapter* m_adapter = nullptr;

    QTimer        m_frameTimer;
    QElapsedTimer m_clock;
    qint64        m_lastNs = 0;
};

class VulkanRendererWidgetWrapper : public QWidget
{
    Q_OBJECT

  public:
    VulkanRendererWidgetWrapper(QWidget* parent = nullptr);
    ~VulkanRendererWidgetWrapper();

    std::optional<Event> PollEvent();

  protected:
    bool eventFilter(QObject* watched, QEvent* event);
    void resizeEvent(QResizeEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

  private:
    void Initialize();
    void InitWidget();
    void InitSignalSlots();

  private:
    VulkanRendererWidget* m_rendererWidget = nullptr;

    EventQueue m_eventQueue;

    bool   m_leftBtnPressing  = false;
    bool   m_rightBtnPressing = false;
    QPoint m_lastPoint       = QPoint(0, 0);
};