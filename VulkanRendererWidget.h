#pragma once
#include <QWidget>

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

  private:
    VulkanAdapter* m_adapter = nullptr;
};
