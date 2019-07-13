#pragma once

#include <iostream>
#include <map>
#include <string>

// парсер аргументов
class ArgParser {
	std::map<std::string, std::string> args; // словарь аргументов

	void AddArg(const std::string arg); // добавление аргумента

public:
	ArgParser(const std::string& s); // создание из строки

	bool operator[](const std::string& key) const; // проверка наличия аргумента
	std::string Get(const std::string& key, const std::string& def = "") const; // получение значения аргумента
};

// добавление аргумента
void ArgParser::AddArg(const std::string arg) {
	size_t i = 0;
	std::string param = "";
	std::string value = "";

	// получаем параметр
	while (i < arg.length() && arg[i] != '=') {
		param += arg[i];
		i++;
	}

	i++;

	while (i < arg.length()) {
		value += arg[i];
		i++;
	}

	args[param] = value.length() == 0 ? "" : value;
}

// создание из строки
ArgParser::ArgParser(const std::string& s) {
	size_t i = 0;

	while (i < s.length()) {
		std::string arg = "";

		// разбиваем строку по пробелам
		while (i < s.length() && s[i] != ' ') {
			arg += s[i];
			i++;
		}

		// если аргумент считан
		if (arg.length())
			AddArg(arg); // добавляем аргумент

		while (i < s.length() && s[i] == ' ')
			i++;
	}
}

// проверка наличия аргумента
bool ArgParser::operator[](const std::string& key) const {
	return args.find(key) != args.end();
}

// получение значения аргумента
std::string ArgParser::Get(const std::string& key, const std::string& def) const {
	return args.find(key) != args.end() ? args.at(key) : def;
}