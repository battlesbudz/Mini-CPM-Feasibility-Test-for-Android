#include "utf8_snapshot.h"
#include <cassert>
int main() {
    for (const std::string& value : {std::string("\xc4\x80"), std::string("\xe2\x80\x91"), std::string("\xf0\x9f\x98\x80")}) {
        for (size_t split = 1; split < value.size(); ++split) {
            auto a = utf8Snapshot("hello " + value.substr(0, split));
            assert(a.text == "hello " && a.pendingBytes == split && a.invalidBytes == 0);
            auto b = utf8Snapshot("hello " + value);
            assert(b.text == "hello " + value && b.pendingBytes == 0 && b.invalidBytes == 0);
        }
    }
    auto partial = utf8Snapshot(" circrose\xc4");
    assert(partial.text == " circrose" && partial.pendingBytes == 1);
    auto final = utf8Snapshot(" circrose\xc4", true);
    assert(final.invalidBytes == 1 && final.pendingBytes == 0);
    for (const std::string& bad : {std::string("\xc0\xaf"), std::string("\xed\xa0\x80"), std::string("\xf4\x90\x80\x80"), std::string("\x80")}) {
        auto result = utf8Snapshot(bad);
        assert(result.invalidBytes > 0 && result.pendingBytes == 0);
    }
    assert(utf8Snapshot("\xc4" "A").text == "\xef\xbf\xbd" "A");
    assert(utf8Snapshot(std::string("a\0b", 3)).text.size() == 3);
}
