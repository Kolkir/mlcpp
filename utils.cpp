#include "utils.h"

#include <stdexcept>

#include <curl/curl.h>
#include <stdio.h>

namespace utils {

namespace {

size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
  return fwrite(ptr, size, nmemb, stream);
}
}  // namespace

bool DownloadFile(const std::string& url, const std::string& path) {
  CURLcode ret{CURLE_OK};
  CURL* hnd = curl_easy_init();
  if (hnd != nullptr) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (fp != nullptr) {
      curl_easy_setopt(hnd, CURLOPT_URL, url.c_str());
      curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION, write_data);
      curl_easy_setopt(hnd, CURLOPT_WRITEDATA, fp);
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
