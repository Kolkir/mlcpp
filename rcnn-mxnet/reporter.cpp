#include "reporter.h"

#include <ncurses.h>

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
  if (!stdout_) {
    std::thread th([&]() {
      WINDOW* win = nullptr;
      win = initscr();
      erase();
      notimeout(win, true);
      nodelay(win, true);

      auto len = values_.size();
      while (!stop_flag_) {
        for (size_t i = 0; i < len; ++i) {
          if (!stdout_) {
            mvprintw(static_cast<int>(i), 0, "%s: %f\n",
                     descriptions_[i].c_str(), static_cast<double>(values_[i]));
          } else {
            std::cout << descriptions_[i].c_str() << ": " << values_[i]
                      << std::endl;
          }
        }
        refresh();
        std::this_thread::sleep_for(interval_);
      }
      endwin();
    });
    print_thread_ = std::move(th);
  }
}

void Reporter::Stop() {
  if (!stdout_) {
    stop_flag_ = true;
    print_thread_.join();
    for (size_t i = 0; i < values_.size(); ++i) {
      std::cout << descriptions_[i].c_str() << ": " << values_[i] << std::endl;
    }
  }
}
