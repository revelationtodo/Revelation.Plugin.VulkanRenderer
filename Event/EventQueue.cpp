#include "EventQueue.h"

bool EventQueue::Empty()
{
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_queue.empty();
}

void EventQueue::Push(const Event& event)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    m_queue.push(event);
}

void EventQueue::Push(Event&& event)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    m_queue.emplace(std::move(event));
}

Event EventQueue::Pop()
{
    std::lock_guard<std::mutex> guard(m_mutex);
    Event                       event = std::move(m_queue.front());
    m_queue.pop();
    return event;
}
