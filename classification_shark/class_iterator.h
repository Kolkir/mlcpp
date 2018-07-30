#ifndef CLASS_ITERATOR_H
#define CLASS_ITERATOR_H

#include <shark/Data/Dataset.h>
#include <limits>

struct ClassIterator {
  // iterator traits
  using difference_type = size_t;
  using value_type = double;
  using pointer = const double*;
  using reference = const double&;
  using iterator_category = std::forward_iterator_tag;
  ClassIterator()
      : label(0),
        data_index(0),
        index(std::numeric_limits<unsigned int>::max()),
        data(nullptr),
        labels(nullptr) {}
  ClassIterator(const shark::Data<shark::RealVector>* data,
                const shark::Data<unsigned int>* labels,
                unsigned int label,
                unsigned int data_index)
      : label(label),
        data_index(data_index),
        index(0),
        data(data),
        labels(labels) {
    bool found = false;
    while (index < labels->numberOfElements()) {
      if (labels->element(index) == label) {
        found = true;
        break;
      }
      ++index;
    }
    if (!found)
      index = std::numeric_limits<unsigned int>::max();
  }
  ClassIterator& operator++() {
    while (index < labels->numberOfElements()) {
      ++index;
      if (index < labels->numberOfElements() &&
          labels->element(index) == label) {
        return *this;
      }
    }
    index = std::numeric_limits<unsigned int>::max();
    return *this;
  }
  double operator*() { return data->element(index)[data_index]; }
  unsigned int label;
  unsigned int data_index;
  unsigned int index;
  const shark::Data<shark::RealVector>* data;
  const shark::Data<unsigned int>* labels;
};

bool operator==(const ClassIterator& a, const ClassIterator& b) {
  return a.index == b.index;
}

bool operator!=(const ClassIterator& a, const ClassIterator& b) {
  return a.index != b.index;
}

#endif // CLASS_ITERATOR_H
