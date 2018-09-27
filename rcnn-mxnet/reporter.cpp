#include "reporter.h"

#include <ncurses.h>
#include <iostream>

Reporter::Reporter(bool stdout,
                   size_t lines_num,
                   std::chrono::milliseconds interval)
    : values_(lines_num),
      descriptions_(lines_num),
      interval_(interval),
      stdout_(stdout) {}

Reporter::~Reporter() {
  Stop();
}

void Reporter::SetLineDescription(size_t index, const std::string& desc) {
  descriptions_.at(index) = desc;
}

void Reporter::Start() {
  std::thread th([&]() {
    WINDOW* win = nullptr;
    if (!stdout_) {
      win = initscr();
      erase();
      notimeout(win, true);
      nodelay(win, true);
    }
    auto len = values_.size();
    while (!stop_flag_) {
      for (size_t i = 0; i < len; ++i) {
        if (!stdout_) {
          mvprintw(static_cast<int>(i), 0, "%s: %f\n", descriptions_[i].c_str(),
                   static_cast<double>(values_[i]));
        } else {
          std::cout << descriptions_[i].c_str() << ": " << values_[i]
                    << std::endl;
        }
      }
      if (!stdout_)
        refresh();
      std::this_thread::sleep_for(interval_);
    }
    if (!stdout_)
      endwin();
  });
  print_thread_ = std::move(th);
}

void Reporter::Stop() {
  stop_flag_ = true;
  print_thread_.join();
}
