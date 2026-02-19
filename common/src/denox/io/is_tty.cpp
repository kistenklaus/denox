#include "denox/io/is_tty.hpp"

#if defined(_WIN32)
    #include <io.h>
#else
    #include <unistd.h>
    #include <cstdio>
#endif

namespace denox::io {

bool stderr_is_tty() noexcept {

#if defined(_WIN32)

    // You explicitly said returning false is fine.
    return false;

#else

    return ::isatty(fileno(stderr));

#endif

}

}
