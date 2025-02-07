#ifndef
#define LAP_HPP

#include <string>

std::string return_ascii() {
    std::string allowed_chars;
    for (int i=97; i<123; i++) {
        allowed_chars += char(i);
        allowed_chars += char(i-32);=
    }
    for (auto c : allowed_chars) {
        std::cout << c << "\n";
    }
}

#endif