#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>

// Define your global debug level here
#define DEBUG_LEVEL 0

// Debug print macro that checks the current debug level
// Usage: DEBUG_PRINT(2, "Error with value: ", value);
#define DEBUG_PRINT(level, ...) \
    do { \
        if (level <= DEBUG_LEVEL) { \
            std::cout << __VA_ARGS__; \
        } \
    } while (0)

#endif // DEBUG_H
