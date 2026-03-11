#pragma once

#include <cstdio>

#ifndef NX_PRINT
#define NX_PRINT(fmt, ...) std::fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif
