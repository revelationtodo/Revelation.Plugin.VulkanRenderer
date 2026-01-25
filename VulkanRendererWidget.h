#pragma once
#include <QWidget>
#include <QTimer>
#include <QElapsedTimer>

class IRevelationInterface;
class VulkanAdapter;

class VulkanRendererWidget : public QWidget
{
    Q_OBJECT

  public:
    VulkanRendererWidget(IRevelationInterface* intf, QWidget* parent = nullptr);
    ~VulkanRendererWidget();

  private:
    void Initialize();
    void InitWidget();
    void InitSignalSlots();

  private slots:
    void TriggerTick();

  private:
    VulkanAdapter* m_adapter = nullptr;

    QTimer        m_frameTimer;
    QElapsedTimer m_clock;
    qint64        m_lastNs = 0;
};
