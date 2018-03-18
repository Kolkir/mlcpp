#ifndef UTILS_H
#define UTILS_H

#include <string>

namespace utils {
bool DownloadFile(const std::string& url, const std::string& path);

template <typename I>
struct iter {
  iter(I iterator) : base_iterator(iterator), index(0) {}
  bool operator!=(const iter& other) const {
    return base_iterator != other.base_iterator;
  }
  auto operator*() const { return std::make_pair(index, *base_iterator); }
  const iter& operator++() {
    ++index;
    ++base_iterator;
    return *this;
  }
  I base_iterator;
  size_t index;
};

template <typename C>
struct enumerate {
  enumerate(C& container) : container(container) {}
  auto begin() { return iter<typename C::iterator>(std::begin(container)); }
  auto end() { return iter(std::end(container)); }
  C& container;
};

}  // namespace utils

#endif  // UTILS_H
