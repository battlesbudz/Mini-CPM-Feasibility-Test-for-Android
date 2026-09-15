#pragma once
#include <string>
#include <cstddef>

// Token pieces are bytes, not necessarily complete Unicode characters.
struct Utf8Snapshot {
    std::string text;
    size_t pendingBytes = 0;
    size_t invalidBytes = 0;
};
inline Utf8Snapshot utf8Snapshot(const std::string& bytes, bool final = false) {
    Utf8Snapshot out;
    for (size_t i = 0; i < bytes.size();) {
        auto c = static_cast<unsigned char>(bytes[i]);
        size_t n = c < 0x80 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 :
                   c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        bool valid = n != 0;
        for (size_t j = 1; valid && j < n && i + j < bytes.size(); ++j) {
            auto b = static_cast<unsigned char>(bytes[i + j]);
            valid = b >= 0x80 && b <= 0xbf;
            if (j == 1) valid = valid && !(c == 0xe0 && b < 0xa0) &&
                !(c == 0xed && b >= 0xa0) && !(c == 0xf0 && b < 0x90) &&
                !(c == 0xf4 && b >= 0x90);
        }
        if (valid && i + n > bytes.size() && !final) {
            out.pendingBytes = bytes.size() - i;
            break;
        }
        if (valid && i + n <= bytes.size()) {
            out.text.append(bytes, i, n);
            i += n;
        } else {
            out.text += "\xef\xbf\xbd";
            ++out.invalidBytes;
            ++i;
        }
    }
    return out;
}
