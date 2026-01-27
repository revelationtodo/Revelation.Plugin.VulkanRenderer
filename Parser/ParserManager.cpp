#include "ParserManager.h"
#include "WavefrontObjParser.h"
#include <filesystem>

ParserManager::ParserManager()
{
    WavefrontObjParser* objParser = new WavefrontObjParser;
    ParserDesc          objParserDesc{.extensions = {".obj"}};
    RegisterParser(objParser, objParserDesc);
}

ParserManager::~ParserManager()
{
    for (const auto& [parser, _] : m_parsers)
    {
        delete parser;
    }
    m_parsers.clear();
}

void ParserManager::RegisterParser(IParser* parser, const ParserDesc& desc)
{
    m_parsers.emplace_back(parser, desc);
}

bool ParserManager::Parse(const std::string& file, Model& model)
{
    std::filesystem::path filePath = std::filesystem::path(file);
    if (!std::filesystem::exists(filePath))
    {
        return false;
    }

    for (const auto& [parser, desc] : m_parsers)
    {
        if (!desc.extensions.count(filePath.extension().string()))
        {
            continue;
        }

        if (parser->CanParse(file))
        {
            parser->Parse(file, model);
            return true;
        }
    }
    return false;
}
