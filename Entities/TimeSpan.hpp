#pragma once

#include <iostream>
#include <iomanip>

// временной промежуток
class TimeSpan {
	int h; // часы
	int m; // минуты
	int s; // секунды
	int ms; // миллисекунды

public:
	TimeSpan(long ms); // создание из миллисекунд

	friend std::ostream& operator<<(std::ostream& os, const TimeSpan &ts); // вывод в поток
};

// создание из миллисекунд
TimeSpan::TimeSpan(long ms) {
	this->ms = ms % 1000;

	s = ms / 1000;
	m = s / 60;
	h = m / 60;

	s %= 60;
	m %= 60;
}

// вывод в поток
std::ostream& operator<<(std::ostream& os, const TimeSpan &ts) {
	os << ts.h << ":"; // выводим часы
	os << std::setw(2) << std::setfill('0') << ts.m << ":"; // выводим минуты
	os << std::setw(2) << std::setfill('0') << ts.s << "."; // выводим секунды
	os << std::setw(3) << ts.ms; // выводим миллисекунды

	return os;
}