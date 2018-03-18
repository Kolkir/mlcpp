#include "utils.h"

#include <stdexcept>

#include <curl/curl.h>
#include <stdio.h>

namespace utils {

bool DownloadFile(const std::string& url, const std::string& path) {
  CURLcode ret{CURLE_OK};
  CURL* hnd = curl_easy_init();
  if (hnd != nullptr) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (fp != nullptr) {
      curl_easy_setopt(hnd, CURLOPT_URL, url.c_str());
      curl_easy_setopt(hnd, CURLOPT_TCP_KEEPALIVE, fp);
      curl_easy_setopt(hnd, CURLOPT_FOLLOWLOCATION, 1L);
      curl_easy_setopt(
          hnd, CURLOPT_NOSIGNAL,
          1);  // Prevent "longjmp causes uninitialized stack frame" bug
      curl_easy_setopt(hnd, CURLOPT_ACCEPT_ENCODING, "deflate");
      curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION, nullptr);
      curl_easy_setopt(hnd, CURLOPT_WRITEDATA, fp);
      // curl_easy_setopt(hnd, CURLOPT_VERBOSE, 1L);
      ret = curl_easy_perform(hnd);
    } else {
      ret = CURLE_FAILED_INIT;
    }
  } else {
    return false;
  }
  curl_easy_cleanup(hnd);
  return ret == CURLE_OK;
}
}  // namespace utils
