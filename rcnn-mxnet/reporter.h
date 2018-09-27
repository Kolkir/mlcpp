#ifndef REPORTER_H
#define REPORTER_H

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

class Reporter {
 public:
  Reporter(bool stdout, size_t lines_num, std::chrono::milliseconds interval);
  ~Reporter();
  Reporter(const Reporter&) = delete;
  Reporter& operator=(const Reporter&) = delete;

  void SetLineDescription(size_t index, const std::string& desc);

  template <typename T>
  void SetLineValue(size_t index, T value) {
    values_[index] = static_cast<float>(value);
  }

  void Start();
  void Stop();

 private:
  std::vector<std::atomic<float>> values_;
  std::vector<std::string> descriptions_;
  std::atomic_bool stop_flag_{false};
  std::thread print_thread_;
  std::chrono::milliseconds interval_;
  bool stdout_{false};
};

#endif  // REPORTER_H
