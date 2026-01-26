#pragma once
#include <QWindow>
#include <QWidget>
#include <QTimer>
#include <QElapsedTimer>
#include <vector>
#include <filesystem>

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
    bool     IsResized();

  protected:
    void resizeEvent(QResizeEvent* event) override;

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

    bool m_resized = false;
};

class VulkanRendererWidgetWrapper : public QWidget
{
    Q_OBJECT

  public:
    VulkanRendererWidgetWrapper(QWidget* parent = nullptr);
    ~VulkanRendererWidgetWrapper();

  protected:
    bool eventFilter(QObject* watched, QEvent* event);
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;

  private:
    void Initialize();
    void InitWidget();
    void InitSignalSlots();

  signals:
    void FileDropped(const std::vector<std::filesystem::path>& paths);

  private:
    VulkanRendererWidget* m_rendererWidget = nullptr;
};