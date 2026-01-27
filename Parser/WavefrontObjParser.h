#pragma once
#include "IParser.h"

class WavefrontObjParser : public IParser
{
  public:
    WavefrontObjParser();
    ~WavefrontObjParser();

    virtual bool CanParse(const std::string& file) override;
    virtual bool Parse(const std::string& file, Model& model) override;
};
