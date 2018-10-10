#include "mxutils.h"
#include <iostream>

void CheckMXnetError(const char* state) {
  auto* err = MXGetLastError();
  if (err && err[0] != 0) {
    std::cout << "MXNet unhandled error in " << state << " : " << err
              << std::endl;
    exit(-1);
  }
}
