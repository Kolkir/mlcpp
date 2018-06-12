// third party includes
#include <shogun/base/init.h>

// stl includes
#include <algorithm>
#include <experimental/filesystem>
#include <iostream>
#include <random>
#include <string>

// application includes
#include "../ioutils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;

int main() {
  shogun::init_shogun_with_defaults();

  shogun::exit_shogun();
  return 0;
}
