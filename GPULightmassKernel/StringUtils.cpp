#include "StringUtils.h"
#include <Windows.h>

namespace RStringUtils
{

std::wstring WidenFromUTF8(const std::string& str)
{
	int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), str.length(), NULL, 0);
	std::wstring wstr(size_needed, 0);
	MultiByteToWideChar(CP_UTF8, 0, str.c_str(), str.length(), &wstr[0], size_needed);

	return wstr;
}

std::wstring Widen(const std::string& str)
{
	int size_needed = MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.length(), NULL, 0);
	std::wstring wstr(size_needed, 0);
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.length(), &wstr[0], size_needed);

	return wstr;
}

std::string Narrow(const std::wstring& wstr)
{
	int size_needed = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.length(), NULL, 0, NULL, NULL);
	std::string str(size_needed, 0);
	WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.length(), &str[0], size_needed, NULL, NULL);

	return str;
}

std::string GetFileNameFromPath(std::string path)
{
	for (int i = path.length() - 1; i >= 0; i--)
	{
		if (path[i] == '/' || path[i] == '\\')
			return path.substr(i + 1);

	}

	return path; //û��б�ܣ���������
}

std::wstring GetFileNameFromPath(std::wstring path)
{
	for (int i = path.length() - 1; i >= 0; i--)
	{
		if (path[i] == L'/' || path[i] == L'\\')
			return path.substr(i + 1);

	}

	return path; //û��б�ܣ���������
}

std::string GetDirFromPath(std::string path) //����ֵ�л����б��
{
	for (int i = path.length() - 1; i >= 0; i--)
	{
		if (path[i] == '/' || path[i] == '\\')
			return path.substr(0, i + 1);

	}

	return ""; //û��б�ܣ�ֻ���ļ��������ؿ�
}

std::wstring GetDirFromPath(std::wstring path) //����ֵ�л����б��
{
	for (int i = path.length() - 1; i >= 0; i--)
	{
		if (path[i] == L'/' || path[i] == L'\\')
			return path.substr(0, i + 1);

	}

	return L""; //û��б�ܣ�ֻ���ļ��������ؿ�
}

bool FileExistTest(const std::string& name) {
	if (FILE *file = fopen(name.c_str(), "r")) {
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}

}
