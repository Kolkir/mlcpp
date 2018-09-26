#include "reporter.h"

#include <ncurses.h>

Reporter::Reporter(size_t lines_num, std::chrono::milliseconds interval)
    : values_(lines_num), descriptions_(lines_num), interval_(interval) {}

Reporter::~Reporter() {
  Stop();
}

void Reporter::SetLineDescription(size_t index, const std::string& desc) {
  descriptions_.at(index) = desc;
}

void Reporter::Start() {
  std::thread th([&]() {
    auto win = initscr();
    erase();
    notimeout(win, true);
    nodelay(win, true);

    auto len = values_.size();
    while (!stop_flag_) {
      for (size_t i = 0; i < len; ++i) {
        mvprintw(static_cast<int>(i), 0, "%s: %f\n", descriptions_[i].c_str(),
                 static_cast<double>(values_[i]));
      }
      refresh();
      std::this_thread::sleep_for(interval_);
    }
    endwin();
  });
  print_thread_ = std::move(th);
}

void Reporter::Stop() {
  stop_flag_ = true;
  print_thread_.join();
}
