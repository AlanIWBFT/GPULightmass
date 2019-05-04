#pragma once
#include <string>

namespace RStringUtils
{

std::wstring WidenFromUTF8(const std::string& str);

std::wstring Widen(const std::string& str);

std::string Narrow(const std::wstring& wstr);

std::string GetFileNameFromPath(std::string path);

std::wstring GetFileNameFromPath(std::wstring path);

std::string GetDirFromPath(std::string path);

std::wstring GetDirFromPath(std::wstring path);

bool FileExistTest(const std::string& name);

}
