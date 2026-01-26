#pragma once
#include <QWindow>
#include <QWidget>
#include <QTimer>
#include <QElapsedTimer>

class IRevelationInterface;
class VulkanAdapter;

class VulkanRendererWidget : public QWindow
{
    Q_OBJECT

  public:
    VulkanRendererWidget(IRevelationInterface* intf, QWidget* parent = nullptr);
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
