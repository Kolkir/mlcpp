#ifndef IOUTILS_H
#define IOUTILS_H

namespace ioutils {
/*
 * from:
 * http://shitalshah.com/p/writing-generic-container-function-in-c11/
 * https://raw.githubusercontent.com/louisdx/cxx-prettyprint/master/prettyprint.hpp
 * also see https://gist.github.com/louisdx/1076849
 */

namespace detail {
// SFINAE type trait to detect whether T::const_iterator exists.

struct sfinae_base {
  using yes = char;
  using no = yes[2];
};

template <typename T>
struct has_const_iterator : private sfinae_base {
 private:
  template <typename C>
  static yes& test(typename C::const_iterator*);
  template <typename C>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
  using type = T;

  void dummy();  // for GCC to supress -Wctor-dtor-privacy
};

template <typename T>
struct not_expression : private sfinae_base {
 private:
  template <typename C>
  static no& test(typename C::expression_tag*);
  template <typename C>
  static yes& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
  using type = T;

  void dummy();  // for GCC to supress -Wctor-dtor-privacy
};

template <typename T>
struct has_printable_value_type : private sfinae_base {
 private:
  template <typename C>
  static yes& test(typename std::enable_if<
                   std::is_fundamental<typename C::value_type>::value>::type*);
  template <typename C>
  static yes& test(
      typename std::enable_if<
          std::is_same<typename C::value_type, std::string>::value>::type*);
  template <typename C>
  static yes& test(
      typename std::enable_if<
          std::is_same<typename C::value_type, std::wstring>::value>::type*);
  template <typename C>
  static yes& test(
      typename std::enable_if<
          std::is_same<typename C::value_type, const char*>::value>::type*);
  template <typename C>
  static yes& test(typename std::enable_if<
                   std::is_same<typename C::value_type, char*>::value>::type*);
  template <typename C>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
  using type = T;

  void dummy();  // for GCC to supress -Wctor-dtor-privacy
};

template <typename T>
struct has_begin_end : private sfinae_base {
 private:
  template <typename C>
  static yes&
  f(typename std::enable_if<
      std::is_same<decltype(static_cast<typename C::const_iterator (C::*)()
                                            const>(&C::begin)),
                   typename C::const_iterator (C::*)() const>::value>::type*);

  template <typename C>
  static no& f(...);

  template <typename C>
  static yes& g(typename std::enable_if<
                std::is_same<decltype(static_cast<typename C::const_iterator (
                                          C::*)() const>(&C::end)),
                             typename C::const_iterator (C::*)() const>::value,
                void>::type*);

  template <typename C>
  static no& g(...);

 public:
  static bool const beg_value = sizeof(f<T>(nullptr)) == sizeof(yes);
  static bool const end_value = sizeof(g<T>(nullptr)) == sizeof(yes);

  void dummy();  // for GCC to supress -Wctor-dtor-privacy
};

}  // namespace detail

template <typename T>
struct is_container : public std::integral_constant<
                          bool,
                          detail::has_const_iterator<T>::value &&
                              detail::has_begin_end<T>::beg_value &&
                              detail::has_begin_end<T>::end_value &&
                              detail::has_printable_value_type<T>::value &&
                              detail::not_expression<T>::value> {};

template <>
struct is_container<std::string> : std::false_type {};

}  // namespace ioutils

template <typename C,
          class = typename std::enable_if<
              ioutils::is_container<typename std::decay<C>::type>::value>::type>
std::ostream& operator<<(std::ostream& out, C&& container) {
  out << '{';
  auto i = std::begin(container);
  auto e = std::end(container);
  auto l = std::prev(std::end(container));
  for (; i != e; ++i) {
    out << *i;
    if (i != l)
      out << ", ";
  }
  out << '}';
  return out;
}

#endif  // IOUTILS_H
