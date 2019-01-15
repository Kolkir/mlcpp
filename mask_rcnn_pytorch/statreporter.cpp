#include "statreporter.h"
#include <iomanip>
#include <iostream>

StatReporter::StatReporter(uint32_t epochs_num,
                           uint32_t train_steps_num,
                           uint32_t val_steps_num)
    : epochs_num_(epochs_num),
      train_steps_num_(train_steps_num),
      val_steps_num_(val_steps_num) {
  print_thread = std::thread([this]() { this->PrintLoop(); });
}

StatReporter::~StatReporter() {
  Stop();
}

void StatReporter::ReportTrainStep(uint32_t i, const LossStat& stat) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    train_step_ = i;
    train_stat_ = stat;
    print_state_ = PrintState::ReportTrainStep;
  }
  state_cv_.notify_one();
}

void StatReporter::ReportValidationStep(uint32_t i, const LossStat& stat) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    train_step_ = i;
    valid_stat_ = stat;
    print_state_ = PrintState::ReportValidationStep;
  }
  state_cv_.notify_one();
}

void StatReporter::StartEpoch(uint32_t i, double learning_rate) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    epoch_ = i;
    learning_rate_ = learning_rate;
    print_state_ = PrintState::StartEpoch;
  }
  state_cv_.notify_one();
}

void StatReporter::ReportEpoch(const LossStat& train_stat,
                               const LossStat& valid_stat) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    train_stat_ = train_stat;
    valid_stat_ = valid_stat;
    print_state_ = PrintState::ReportEpoch;
  }
  state_cv_.notify_one();
}

void StatReporter::Stop() {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    print_state_ = PrintState::Stop;
  }
  state_cv_.notify_one();
  print_thread.join();
}

void StatReporter::PrintLoop() {
  bool done = false;
  while (!done) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    state_cv_.wait(lock, [this] { return print_state_ != PrintState::Idle; });
    switch (print_state_) {
      case PrintState::StartEpoch:
        std::cerr << "\nEpoch " << epoch_ << "/" << epochs_num_ << "\n";
        std::cerr << "\tlearning rate " << learning_rate_ << "\n";
        print_state_ = PrintState::Idle;
        break;
      case PrintState::ReportEpoch:
        std::cerr << "\tTraining losses :\n";
        PrintLoss(std::cerr, train_stat_);
        std::cerr << "\tValidation losses :\n";
        PrintLoss(std::cerr, valid_stat_);
        print_state_ = PrintState::Idle;
        break;
      case PrintState::ReportTrainStep:
        ClearStepScreen();
        std::cerr << "\tTrain step " << train_step_ << "/" << train_steps_num_;
        PrintLossSmall(std::cerr, train_stat_);
        std::cerr << "\r";
        print_state_ = PrintState::Idle;
        break;
      case PrintState::ReportValidationStep:
        ClearStepScreen();
        std::cerr << "\tValidation step " << train_step_ << "/"
                  << val_steps_num_;
        PrintLossSmall(std::cerr, valid_stat_);
        std::cerr << "\r";
        print_state_ = PrintState::Idle;
        break;
      case PrintState::Idle:
        // ignore
        break;
      case PrintState::Stop:
        done = true;
        break;
    }
  }
}

void StatReporter::ClearStepScreen() {}

void StatReporter::PrintLoss(std::ostream& out, const LossStat& stat) {
  out << "\t\t" << std::setw(20) << "loss sum : " << stat.loss << "\n";
  out << "\t\t" << std::setw(20) << "loss_rpn_class : " << stat.loss_rpn_class
      << "\n";
  out << "\t\t" << std::setw(20) << "loss_rpn_bbox : " << stat.loss_rpn_bbox
      << "\n";
  out << "\t\t" << std::setw(20)
      << "loss_mrcnn_class : " << stat.loss_mrcnn_class << "\n";
  out << "\t\t" << std::setw(20) << "loss_mrcnn_bbox : " << stat.loss_mrcnn_bbox
      << "\n";
  out << "\t\t" << std::setw(20) << "loss_mrcnn_mask : " << stat.loss_mrcnn_mask
      << "\n";
}

void StatReporter::PrintLossSmall(std::ostream& out, const LossStat& stat) {
  out << std::fixed << std::setprecision(3);
  out << " sum : " << stat.loss;
  out << " rpn_class : " << stat.loss_rpn_class;
  out << " rpn_bbox : " << stat.loss_rpn_bbox;
  out << " mrcnn_class : " << stat.loss_mrcnn_class;
  out << " mrcnn_bbox : " << stat.loss_mrcnn_bbox;
  out << " mrcnn_mask : " << stat.loss_mrcnn_mask;
  out << "              ";
}
