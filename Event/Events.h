#pragma once

enum class EventType
{
    None,
    ResizeEvent,
    DropEvent,
    MouseEvent
};

struct Event
{
    EventType type = EventType::None;
    std::any  data = {};
};

struct ResizeEventData
{
};

struct DropEventData
{
    std::string file = "";
};

enum class MouseEventType
{
    None,
    Press,
    Release,
    DoubleClick,
    Move,
    Wheel,
};

enum class MouseBtnType
{
    None,
    Left,
    Right,
};

struct MouseEventData
{
    MouseEventType event           = MouseEventType::None;
    MouseBtnType   btn             = MouseBtnType::None;
    int            deltaX          = 0;
    int            deltaY          = 0;
    bool           leftBtnPressed  = false;
    bool           rightBtnPressed = false;
};