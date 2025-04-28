#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <string>

/**
 * @brief Pauses program execution until the user presses Enter
 *
 * This function reads a line from standard input and does nothing with it,
 * effectively pausing the program until the user presses Enter.
 */
inline void console_pause()
{
    std::string dummy;
    std::cout << "Press Enter to continue...";
    std::getline(std::cin, dummy);
}

#endif // HELPER_H