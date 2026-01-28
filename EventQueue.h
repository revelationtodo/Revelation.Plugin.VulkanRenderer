#pragma once
#include <any>
#include <mutex>
#include <queue>
#include "Event.h"

class EventQueue
{
  public:
    EventQueue()  = default;
    ~EventQueue() = default;

    bool  Empty();
    void  Push(const Event& event);
    void  Push(Event&& event);
    Event Pop();

  private:
    std::mutex        m_mutex;
    std::queue<Event> m_queue;
};
