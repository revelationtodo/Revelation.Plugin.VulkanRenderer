#pragma once
#include "IParser.h"
#include <vector>

class ParserManager
{
  public:
    ParserManager();
    ~ParserManager();

    void RegisterParser(IParser* parser, const ParserDesc& desc);

    bool Parse(const std::string& file, Model& model);

  private:
    std::vector<std::pair<IParser*, ParserDesc>> m_parsers;
};